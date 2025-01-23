from llama_index.core import SimpleDirectoryReader
from entity_extractor import EntityRelationshipExtractor
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List, Dict
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()

def load_documents(data_dir: str) -> List[str]:
    """Load documents and return their text content."""
    print(f"\nLoading documents from {data_dir}")
    documents = SimpleDirectoryReader(data_dir).load_data()
    return [doc.text for doc in documents]

def process_documents(texts: List[str], batch_size: int = 5) -> Dict:
    """Process documents in batches and extract entities and relationships."""
    # Initialize the extractor with caching enabled
    extractor = EntityRelationshipExtractor(
        llm_model="llama3.3:70b",
        base_url="http://localhost:11434",
        batch_size=batch_size,
        cache_enabled=True
    )
    
    start_time = time.time()
    
    # Process all documents
    print("\nExtracting entities and relationships...")
    entities, relationships = extractor.process_documents(texts, show_progress=True)
    
    # Save the complete knowledge graph
    output_file = "knowledge_graph.json"
    extractor.save_to_file(output_file)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print statistics
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Total entities found: {len(entities)}")
    print(f"Total relationships found: {len(relationships)}")
    print(f"\nResults saved to {output_file}")
    
    # Print entity breakdown
    print("\nEntities by type:")
    for entity_type, entities_of_type in extractor.entities_by_type.items():
        print(f"{entity_type}: {len(entities_of_type)} entities")
        for entity in entities_of_type:
            print(f"  - {entity.name}")
    
    # Print relationship summary
    print("\nRelationships found:")
    for rel in relationships:
        print(f"{rel.source.name} --[{rel.relation_type}]--> {rel.target.name}")
    
    return {
        'processing_time': processing_time,
        'total_entities': len(entities),
        'total_relationships': len(relationships),
        'entities_by_type': {
            entity_type: len(entities_of_type)
            for entity_type, entities_of_type in extractor.entities_by_type.items()
        }
    }

def main():
    # Configuration
    data_dir = "./data/blackrock"
    batch_size = 5
    
    try:
        # Load documents
        texts = load_documents(data_dir)
        print(f"Loaded {len(texts)} documents")
        
        # Process documents and get statistics
        stats = process_documents(texts, batch_size)
        
        # Save processing statistics
        stats_file = "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"\nProcessing statistics saved to {stats_file}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()