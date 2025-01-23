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

    def __hash__(self):
        # Use MD5 hashing of the (lower-cased) name and type to ensure uniqueness
        return int(hashlib.md5(f"{self.name.lower()}:{self.type}".encode()).hexdigest(), 16)

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower() and self.type == other.type


@dataclass
class Relationship:
    """
    Represents a directional relationship between two entities, annotated
    with a relation type (e.g., "BORN_IN").
    """
    source: Entity
    target: Entity
    relation_type: str

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
        cache_enabled: bool = True
    ):
        self.batch_size = batch_size
        self.cache = EntityCache() if cache_enabled else None

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
        Naive method: find the first occurrence of entity_name in text
        and return (start_char, end_char). If not found, returns (-1, -1).
        """
        # You could remove this if you don't need positions.
        lower_text = text.lower()
        lower_entity = entity_name.lower()
        idx = lower_text.find(lower_entity)
        if idx == -1:
            return -1, -1
        return idx, idx + len(entity_name)

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
For each entity, provide a short descriptive type.
Return each in the format:

entity_name|entity_type

Text:
{text}
"""
        response = self.llm.complete(prompt)
        lines = response.text.strip().split('\n')

        extracted_entities = []
        for line in lines:
            if '|' not in line:
                continue
            parts = line.split('|', 1)
            if len(parts) != 2:
                continue
            name, entity_type = [p.strip() for p in parts]

            # Attempt naive span detection
            start_char, end_char = self._find_entity_span(name, text)
            snippet = text[start_char:end_char] if (start_char != -1) else name

            extracted_entities.append(
                Entity(
                    name=name,
                    type=entity_type,
                    start_char=start_char,
                    end_char=end_char,
                    text=snippet
                )
            )

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
            return []

        text_hash = self._get_text_hash(text)
        if self.cache:
            cached_result = self.cache.get(text_hash)
            if cached_result and 'relationships' in cached_result:
                # Rebuild Relationship objects from cache
                relationships = []
                # Rebuild quick lookup by name+type
                ent_map = {(e.name.strip().lower(), e.type.lower()): e for e in entities}
                for rel_data in cached_result['relationships']:
                    source_key = (rel_data['source'].strip().lower(), rel_data['source_type'].strip().lower())
                    target_key = (rel_data['target'].strip().lower(), rel_data['target_type'].strip().lower())
                    if source_key in ent_map and target_key in ent_map:
                        relationships.append(
                            Relationship(
                                source=ent_map[source_key],
                                target=ent_map[target_key],
                                relation_type=rel_data['relation_type']
                            )
                        )
                return relationships

        # Build a prompt listing the entities
        entities_list_str = "\n".join(f" - {e.name} ({e.type})" for e in entities)
        prompt = f"""You are an NLP system that performs relationship extraction from text.
Identify all possible relationships between these entities:

{entities_list_str}

Text:
{text}

Return each relationship in the format:
source_entity|relationship_type|target_entity
"""
        response = self.llm.complete(prompt)
        lines = response.text.strip().split('\n')

        found_relationships = []
        for line in lines:
            if '|' not in line:
                continue
            parts = line.split('|')
            if len(parts) != 3:
                continue
            source_name, rel_type, target_name = [p.strip() for p in parts]

            # Find matching Entities
            source_entity = None
            target_entity = None
            for ent in entities:
                if ent.name.strip().lower() == source_name.lower():
                    source_entity = ent
                if ent.name.strip().lower() == target_name.lower():
                    target_entity = ent

            if source_entity and target_entity:
                found_relationships.append(Relationship(
                    source=source_entity,
                    target=target_entity,
                    relation_type=rel_type
                ))

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
            entities = self.extract_entities(text)
            relationships = self.extract_relationships(text, entities)

            # Collect them in our global store
            for entity in entities:
                self.entities_by_type[entity.type].add(entity)

            for rel in relationships:
                self.relationships.add(rel)

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
