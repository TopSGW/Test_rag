import os
import shutil
import warnings
import re

import kuzu
import nest_asyncio
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.graph_stores.kuzu import KuzuPropertyGraphStore
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.ollama import Ollama
from entity_extractor import EntityRelationshipExtractor
import numpy as np

# Load environment variables
load_dotenv()
SEED = 42
nest_asyncio.apply()

# Set up the embedding model using Ollama
embed_model = OllamaEmbedding(
    model_name="llama3.3:70b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Configure Ollama LLMs
extraction_llm = Ollama(
    model="llama3.3:70b",
    temperature=0.0,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

generation_llm = Ollama(
    model="llama3.3:70b",
    temperature=0.3,
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

def clean_table_name(name: str) -> str:
    """Clean a string to be a valid table name."""
    # Remove any characters that aren't alphanumeric or underscore
    cleaned = re.sub(r'[^\w\s]', '', name)
    # Replace spaces with underscores and convert to uppercase
    cleaned = cleaned.replace(' ', '_').upper()
    # Ensure it starts with a letter (prepend E_ if it doesn't)
    if not cleaned[0].isalpha():
        cleaned = 'E_' + cleaned
    return cleaned

def get_relationship_embedding(rel_type: str) -> np.ndarray:
    """Generate embedding for a relationship type."""
    # Create a descriptive text for the relationship
    rel_text = f"This represents a {rel_type} relationship between entities"
    # Get embedding using the Ollama model
    embedding = embed_model.get_text_embedding(rel_text)
    return np.array(embedding)

# Load the dataset on Larry Fink
original_documents = SimpleDirectoryReader("./data/blackrock").load_data()

# --- Step 1: Chunk and store the vector embeddings in LanceDB ---
shutil.rmtree("./test_lancedb", ignore_errors=True)

vector_store = LanceDBVectorStore(
    uri="./test_lancedb",
    mode="overwrite",
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=32),
        embed_model,
    ],
    vector_store=vector_store,
)
pipeline.run(documents=original_documents)

# Create the vector index
vector_index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
    llm=generation_llm,
)

# --- Step 2: Construct the graph in KùzuDB ---
shutil.rmtree("test_kuzudb", ignore_errors=True)
db = kuzu.Database("test_kuzudb")

warnings.filterwarnings("ignore")

# Initialize the graph store without schema validation
graph_store = KuzuPropertyGraphStore(
    db,
    has_structured_schema=False,  # Disable schema validation
)

# --- Step 3: Use EntityRelationshipExtractor to automatically extract entities and relationships ---

# Initialize the extractor
extractor = EntityRelationshipExtractor(
    llm_model="llama3.3:70b",
    base_url="http://localhost:11434",
    cache_enabled=True
)

# Process all documents to extract entities and relationships
texts = [doc.text for doc in original_documents]
entities, relationships = extractor.process_documents(texts, show_progress=True)

# Open a connection to the database to modify the graph
conn = kuzu.Connection(db)

# Track created entity types and their mappings
created_entity_types = set()
entity_type_mapping = {}

# Create tables and add entities dynamically
for entity in entities:
    original_type = entity.type.upper()
    # Clean the entity type name for use as a table name
    entity_type = clean_table_name(original_type)
    entity_type_mapping[original_type] = entity_type
    
    # Create table if it doesn't exist
    if entity_type not in created_entity_types:
        try:
            conn.execute(f"CREATE NODE TABLE IF NOT EXISTS {entity_type} (id STRING, name STRING, PRIMARY KEY (id))")
            created_entity_types.add(entity_type)
        except Exception as e:
            print(f"Error creating table {entity_type}: {str(e)}")
            continue
    
    # Add entity
    try:
        conn.execute(
            f"""
            MERGE (e:{entity_type} {{id: $name, name: $name}})
            """,
            parameters={"name": entity.name},
        )
    except Exception as e:
        print(f"Error adding entity {entity.name}: {str(e)}")

# Track and create relationship tables
relationship_tables = set()
for rel in relationships:
    source_type = entity_type_mapping.get(rel.source.type.upper())
    target_type = entity_type_mapping.get(rel.target.type.upper())
    if not source_type or not target_type:
        continue
    
    # Clean the relationship type
    relation_type = clean_table_name(rel.relation_type)
    
    # Create relationship table if it doesn't exist
    if relation_type not in relationship_tables:
        try:
            # Most relationships are many-to-many by default
            multiplicity = "MANY_MANY"
            
            # Create relationship table with embedding as VAR_LIST FLOAT
            conn.execute(f"""
                CREATE REL TABLE IF NOT EXISTS {relation_type} (
                    FROM {source_type} TO {target_type},
                    embedding VAR_LIST<FLOAT>,
                    description STRING,
                    {multiplicity}
                )
            """)
            relationship_tables.add(relation_type)
            print(f"Created relationship table {relation_type} with multiplicity {multiplicity}")
        except Exception as e:
            print(f"Error creating relationship table {relation_type}: {str(e)}")
            continue

# Add relationships with embeddings
for rel in relationships:
    source_type = entity_type_mapping.get(rel.source.type.upper())
    target_type = entity_type_mapping.get(rel.target.type.upper())
    if not source_type or not target_type:
        continue
    
    # Clean the relationship type
    relation_type = clean_table_name(rel.relation_type)
    
    try:
        # Generate embedding for the relationship
        embedding = get_relationship_embedding(rel.relation_type)
        description = f"{rel.source.name} {rel.relation_type} {rel.target.name}"
        
        # Create relationship with embedding stored as VAR_LIST
        conn.execute(
            f"""
            MATCH (s:{source_type}), (t:{target_type})
            WHERE s.id = $source_name AND t.id = $target_name
            MERGE (s)-[r:{relation_type} {{
                embedding: $embedding,
                description: $description
            }}]->(t)
            """,
            parameters={
                "source_name": rel.source.name,
                "target_name": rel.target.name,
                "embedding": embedding.tolist(),
                "description": description
            },
        )
            
    except Exception as e:
        print(f"Error adding relationship {rel.relation_type}: {str(e)}")

# Add birth dates for known individuals (this information might not be extractable from text)
person_type = entity_type_mapping.get('PERSON', 'PERSON')
try:
    conn.execute(f"ALTER TABLE {person_type} ADD birth_date STRING")
except RuntimeError:
    pass

birth_dates = {
    "Larry Fink": "1952-11-02",
    "Susan Wagner": "1961-05-26",
    "Robert Kapito": "1957-02-08"
}

for name, date in birth_dates.items():
    try:
        conn.execute(
            f"""
            MERGE (p:{person_type} {{id: $name}})
            ON MATCH SET p.birth_date = $date
            """,
            parameters={"name": name, "date": date},
        )
    except Exception as e:
        print(f"Error adding birth date for {name}: {str(e)}")

# Example query to find semantically similar relationships
print("\nSearching for semantically similar relationships...")
test_rel = "FOUNDED"
test_embedding = get_relationship_embedding(test_rel).tolist()

# Search for similar relationships using VAR_LIST dot product
try:
    result = conn.execute("""
        MATCH (a)-[r]->(b)
        WITH r.description AS desc,
             r.embedding AS emb,
             $test_embedding AS test_emb
        WHERE emb IS NOT NULL
        WITH desc, 
             reduce(dot = 0.0, i IN RANGE(0, size(emb)-1) | 
                dot + emb[i] * test_emb[i]) / 
             (sqrt(reduce(norm1 = 0.0, i IN RANGE(0, size(emb)-1) | 
                norm1 + emb[i] * emb[i])) * 
              sqrt(reduce(norm2 = 0.0, i IN RANGE(0, size(test_emb)-1) | 
                norm2 + test_emb[i] * test_emb[i]))) AS similarity
        WHERE similarity > 0.85
        RETURN desc, similarity
        ORDER BY similarity DESC
        LIMIT 5
    """, parameters={"test_embedding": test_embedding})
    
    while result.has_next():
        row = result.get_next()
        print(f"  - {row[0]} (similarity: {row[1]:.3f})")
except Exception as e:
    print(f"Error searching for similar relationships: {str(e)}")

conn.close()