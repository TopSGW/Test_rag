from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from collections import defaultdict
from tqdm import tqdm
import hashlib
import json
from pathlib import Path
import subprocess
import re


@dataclass
class Entity:
    """
    Represents an extracted entity, holding its name, semantic type,
    character positions, and the slice of text in which it was found.
    """
    name: str
    type: str
    start_char: int
    end_char: int
    text: str

    def __post_init__(self):
        # Clean the name and type
        self.name = self._clean_entity_name(self.name)
        self.type = self._clean_entity_type(self.type)

    def _clean_entity_name(self, name: str) -> str:
        # Remove any notes in parentheses and extra whitespace
        name = re.sub(r'\s*\([^)]*\)', '', name)
        return name.strip()

    def _clean_entity_type(self, type_str: str) -> str:
        # Remove any notes in parentheses
        type_str = re.sub(r'\s*\([^)]*\)', '', type_str)
        # Standardize common type names
        type_map = {
            'PERSON': ['PERSON', 'INDIVIDUAL', 'PEOPLE'],
            'ORGANIZATION': ['ORGANIZATION', 'COMPANY', 'CORPORATION', 'INSTITUTION'],
            'LOCATION': ['LOCATION', 'PLACE', 'CITY', 'STATE', 'COUNTRY'],
            'DATE': ['DATE', 'TIME', 'YEAR'],
            'DEGREE': ['DEGREE', 'QUALIFICATION', 'EDUCATION'],
            'MONETARY_VALUE': ['MONETARY_VALUE', 'MONEY', 'CURRENCY']
        }
        
        type_str = type_str.upper().strip()
        for standard_type, variants in type_map.items():
            if type_str in variants:
                return standard_type
        return type_str

    def __hash__(self):
        # Use MD5 hashing of the (lower-cased) name and type to ensure uniqueness
        return int(hashlib.md5(f"{self.name.lower()}:{self.type}".encode()).hexdigest(), 16)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower() and self.type == other.type

    def __str__(self):
        return f"{self.name} ({self.type})"


@dataclass
class Relationship:
    """
    Represents a directional relationship between two entities, annotated
    with a relation type (e.g., "BORN_IN").
    """
    source: Entity
    target: Entity
    relation_type: str

    def __post_init__(self):
        # Standardize relationship type
        self.relation_type = self._standardize_relation_type(self.relation_type)

    def _standardize_relation_type(self, rel_type: str) -> str:
        # Remove any notes in parentheses and convert to uppercase
        rel_type = re.sub(r'\s*\([^)]*\)', '', rel_type)
        rel_type = rel_type.upper().strip()
        
        # Standardize common relationship types
        relation_map = {
            'FOUNDED': ['FOUNDED', 'ESTABLISHED', 'CREATED', 'STARTED'],
            'WORKS_AT': ['WORKS_AT', 'EMPLOYED_BY', 'WORKS_FOR'],
            'STUDIED_AT': ['STUDIED_AT', 'ATTENDED', 'GRADUATED_FROM'],
            'BORN_IN': ['BORN_IN', 'BIRTHPLACE'],
            'LOCATED_IN': ['LOCATED_IN', 'BASED_IN', 'SITUATED_IN']
        }
        
        for standard_rel, variants in relation_map.items():
            if rel_type in variants:
                return standard_rel
        return rel_type

    def __hash__(self):
        return int(
            hashlib.md5(
                f"{self.source.name.lower()}:{self.relation_type}:{self.target.name.lower()}".encode()
            ).hexdigest(),
            16
        )

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return (
            self.source == other.source 
            and self.target == other.target 
            and self.relation_type == other.relation_type
        )

    def __str__(self):
        return f"{self.source.name} --[{self.relation_type}]--> {self.target.name}"


class EntityCache:
    """
    Simple JSON-based cache to avoid repeated LLM calls for the same text.
    Stores both 'entities' and 'relationships' keyed by an MD5 hash of the text.
    """
    def __init__(self, cache_file: str = "entity_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            with self.cache_file.open('r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with self.cache_file.open('w', encoding='utf-8') as f:
            json.dump(self.cache, f, indent=2)

    def get(self, text_hash: str) -> Optional[Dict]:
        return self.cache.get(text_hash)

    def set(self, text_hash: str, result: Dict):
        self.cache[text_hash] = result
        self.save_cache()


class EntityRelationshipExtractor:
    """
    A pipeline that:
      1) Extracts entities from text (only via LLM).
      2) Extracts relationships between these entities via LLM.
      3) Caches results to avoid repeated extractions.
    """
    def __init__(
        self,
        llm_model: str = "llama3.3:70b",
        base_url: str = "http://localhost:11434",
        batch_size: int = 5,
        cache_enabled: bool = True,
        debug: bool = True
    ):
        self.batch_size = batch_size
        self.cache = EntityCache() if cache_enabled else None
        self.debug = debug

        # LLM for entity and relationship extraction
        self.llm = Ollama(
            model=llm_model,
            temperature=0.0,
            request_timeout=120.0,
            base_url=base_url
        )

        self.embed_model = OllamaEmbedding(
            model_name=llm_model,
            base_url=base_url,
        )

        # In-memory containers for processed data
        self.entities_by_type = defaultdict(set)
        self.relationships = set()

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _find_entity_span(self, entity_name: str, text: str) -> Tuple[int, int]:
        """
        Find the first occurrence of entity_name in text
        and return (start_char, end_char). If not found, returns (-1, -1).
        """
        lower_text = text.lower()
        lower_entity = entity_name.lower()
        idx = lower_text.find(lower_entity)
        if idx == -1:
            return -1, -1
        return idx, idx + len(entity_name)

    def _find_matching_entity(self, name: str, type_hint: str, entities: List[Entity]) -> Optional[Entity]:
        """Find a matching entity with fuzzy name matching."""
        name = name.lower().strip()
        # First try exact match
        for entity in entities:
            if entity.name.lower().strip() == name:
                return entity
        
        # Try partial matches
        for entity in entities:
            if name in entity.name.lower() or entity.name.lower() in name:
                return entity
        
        # Try matching by type if provided
        if type_hint:
            type_hint = type_hint.upper().strip()
            for entity in entities:
                if entity.type == type_hint and (name in entity.name.lower() or entity.name.lower() in name):
                    return entity
        
        return None

    def extract_entities(self, text: str) -> List[Entity]:
        """
        Main method for entity extraction with only the LLM. 
        We:
          1) Check cache.
          2) If no cached result, call the LLM.
          3) Build Entity objects, store in cache.
        """
        text_hash = self._get_text_hash(text)

        if self.cache:
            cached_result = self.cache.get(text_hash)
            if cached_result and 'entities' in cached_result:
                # Return cached Entities
                return [Entity(**entity_data) for entity_data in cached_result['entities']]

        prompt = f"""You are an NLP system. Extract ALL named entities from the text below.
For each entity, provide a short descriptive type (PERSON, ORGANIZATION, LOCATION, DATE, etc.).
Return each in the format:

entity_name|entity_type

Text:
{text}
"""
        response = self.llm.complete(prompt)
        lines = response.text.strip().split('\n')

        extracted_entities = []
        seen_entities = set()  # Track unique entities

        for line in lines:
            if '|' not in line:
                continue
            parts = line.split('|', 1)
            if len(parts) != 2:
                continue
            name, entity_type = [p.strip() for p in parts]

            # Create entity and let post_init clean the name and type
            entity = Entity(
                name=name,
                type=entity_type,
                start_char=-1,
                end_char=-1,
                text=name
            )

            # Only add if we haven't seen this entity before
            entity_key = (entity.name.lower(), entity.type)
            if entity_key not in seen_entities:
                # Try to find the span in the text
                start_char, end_char = self._find_entity_span(entity.name, text)
                if start_char != -1:
                    entity.start_char = start_char
                    entity.end_char = end_char
                    entity.text = text[start_char:end_char]

                extracted_entities.append(entity)
                seen_entities.add(entity_key)

        if self.debug:
            print(f"\nExtracted {len(extracted_entities)} unique entities:")
            for entity in extracted_entities:
                print(f"  - {entity}")

        # Cache the final result
        if self.cache:
            self.cache.set(text_hash, {
                'entities': [
                    {
                        'name': e.name,
                        'type': e.type,
                        'start_char': e.start_char,
                        'end_char': e.end_char,
                        'text': e.text
                    }
                    for e in extracted_entities
                ]
            })

        return extracted_entities

    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships among entities using the LLM only.
        """
        if len(entities) < 2:
            if self.debug:
                print("Not enough entities to extract relationships")
            return []

        text_hash = self._get_text_hash(text)
        if self.cache:
            cached_result = self.cache.get(text_hash)
            if cached_result and 'relationships' in cached_result:
                # Rebuild Relationship objects from cache
                relationships = []
                for rel_data in cached_result['relationships']:
                    source = self._find_matching_entity(rel_data['source'], rel_data['source_type'], entities)
                    target = self._find_matching_entity(rel_data['target'], rel_data['target_type'], entities)
                    
                    if source and target:
                        rel = Relationship(
                            source=source,
                            target=target,
                            relation_type=rel_data['relation_type']
                        )
                        relationships.append(rel)
                        if self.debug:
                            print(f"Found cached relationship: {rel}")
                return relationships

        # Build a prompt listing the entities
        entities_list_str = "\n".join(f" - {e.name} ({e.type})" for e in entities)
        prompt = f"""You are an NLP system that performs relationship extraction from text.
Your task is to identify relationships between the entities listed below.
Please find all explicit and implicit relationships. 

Entities:
{entities_list_str}

Text:
{text}

Instructions:
1. Analyze the text carefully to find relationships between the listed entities.
2. Only extract relationships that are supported by the text.
3. Use clear and consistent relationship types (e.g., FOUNDED, WORKS_AT, STUDIED_AT, BORN_IN).
4. Return each relationship in the format: source_entity|relationship_type|target_entity

Example formats:
John Smith|FOUNDED|Tech Corp
Jane Doe|WORKS_AT|Tech Corp
John Smith|GRADUATED_FROM|Harvard University

Return only the relationships, one per line:
"""
        response = self.llm.complete(prompt)
        lines = response.text.strip().split('\n')

        if self.debug:
            print("\nLLM Response for relationships:")
            print(response.text.strip())

        found_relationships = []
        for line in lines:
            if '|' not in line:
                if self.debug:
                    print(f"Skipping invalid line (no delimiter): {line}")
                continue
            
            parts = line.split('|')
            if len(parts) != 3:
                if self.debug:
                    print(f"Skipping invalid line (wrong number of parts): {line}")
                continue
            
            source_name, rel_type, target_name = [p.strip() for p in parts]

            # Find matching Entities using fuzzy matching
            source_entity = self._find_matching_entity(source_name, "", entities)
            target_entity = self._find_matching_entity(target_name, "", entities)

            if source_entity and target_entity:
                rel = Relationship(
                    source=source_entity,
                    target=target_entity,
                    relation_type=rel_type
                )
                found_relationships.append(rel)
                if self.debug:
                    print(f"Found relationship: {rel}")
            else:
                if self.debug:
                    print(f"Could not find entities for relationship: {line}")
                    if not source_entity:
                        print(f"Missing source entity: {source_name}")
                    if not target_entity:
                        print(f"Missing target entity: {target_name}")

        if self.debug:
            print(f"\nExtracted {len(found_relationships)} relationships:")
            for rel in found_relationships:
                print(f"  - {rel}")

        if self.cache:
            self.cache.set(text_hash, {
                'relationships': [
                    {
                        'source': rel.source.name,
                        'source_type': rel.source.type,
                        'target': rel.target.name,
                        'target_type': rel.target.type,
                        'relation_type': rel.relation_type
                    }
                    for rel in found_relationships
                ]
            })

        return found_relationships

    def process_documents(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> Tuple[Set[Entity], Set[Relationship]]:
        """
        Process multiple documents and accumulate a global
        set of unique entities and relationships.
        """
        if show_progress:
            texts = tqdm(texts, desc="Processing documents")

        for text in texts:
            try:
                entities = self.extract_entities(text)
                if self.debug:
                    print(f"\nProcessing text of length {len(text)} characters")

                relationships = self.extract_relationships(text, entities)

                # Collect them in our global store
                for entity in entities:
                    self.entities_by_type[entity.type].add(entity)

                for rel in relationships:
                    self.relationships.add(rel)

            except Exception as e:
                print(f"Error processing text: {str(e)}")
                continue

        # Combine all entities from the dictionary into a single set
        all_entities = {
            entity
            for entity_set in self.entities_by_type.values()
            for entity in entity_set
        }
        return all_entities, self.relationships

    def get_entity_graph(self, text: str) -> Dict:
        """
        Generate a graph representation of entities and relationships from a single text.
        """
        entities = self.extract_entities(text)
        relationships = self.extract_relationships(text, entities)

        graph = {
            'entities': defaultdict(list),
            'relationships': []
        }

        for entity in entities:
            graph['entities'][entity.type].append({
                'name': entity.name,
                'text': entity.text,
                'position': (entity.start_char, entity.end_char)
            })

        for rel in relationships:
            graph['relationships'].append({
                'source': {
                    'name': rel.source.name,
                    'type': rel.source.type
                },
                'target': {
                    'name': rel.target.name,
                    'type': rel.target.type
                },
                'type': rel.relation_type
            })

        # Convert defaultdict to a regular dict
        graph['entities'] = dict(graph['entities'])
        return dict(graph)

    def save_to_file(self, output_file: str = "extracted_knowledge_graph.json"):
        """
        Save the entire knowledge graph (from all processed docs) to a file.
        """
        output = {
            'entities': {
                entity_type: [
                    {
                        'name': entity.name,
                        'text': entity.text,
                        'position': (entity.start_char, entity.end_char)
                    }
                    for entity in entities
                ]
                for entity_type, entities in self.entities_by_type.items()
            },
            'relationships': [
                {
                    'source': {
                        'name': rel.source.name,
                        'type': rel.source.type
                    },
                    'target': {
                        'name': rel.target.name,
                        'type': rel.target.type
                    },
                    'type': rel.relation_type
                }
                for rel in self.relationships
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)