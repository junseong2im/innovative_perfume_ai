"""
Data Loader Utility for Fragrance AI Database Files
Handles loading and validation of various JSON databases
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Enumeration of available database types"""
    MUSIC_TO_FRAGRANCE = "music_to_fragrance_database.json"
    TEXTURE_TO_FRAGRANCE = "texture_to_fragrance_database.json"
    FRAGRANCE_NOTES = "comprehensive_fragrance_notes_database.json"
    FRAGRANCE_RECIPES = "fragrance_recipes_database.json"
    PERFUME_RULES = "perfume_blending_rules.json"
    MASTER_KNOWLEDGE = "master_perfumer_knowledge_base.json"
    COLOR_TO_FRAGRANCE = "color_to_fragrance_mapping_system.json"
    ARCHITECTURE_TO_FRAGRANCE = "architecture_art_to_fragrance_system.json"
    MASTERPIECE_TO_FRAGRANCE = "masterpiece_to_fragrance_database.json"
    SPATIAL_TO_SCENT = "spatial_to_scent_conversion_algorithm.json"


@dataclass
class LoadResult:
    """Result of database loading operation"""
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]


class DataLoader:
    """Universal data loader for fragrance databases"""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize DataLoader

        Args:
            base_path: Base directory path for data files. If None, uses current directory/data
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            # Default to data directory relative to this script
            self.base_path = Path(__file__).parent

        if not self.base_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.base_path}")

        self._cache = {}
        logger.info(f"DataLoader initialized with base path: {self.base_path}")

    def load_database(self, db_type: Union[DatabaseType, str],
                     use_cache: bool = True) -> LoadResult:
        """
        Load a specific database

        Args:
            db_type: Type of database to load (DatabaseType enum or string filename)
            use_cache: Whether to use cached data if available

        Returns:
            LoadResult object with success status, data, and metadata
        """
        try:
            # Handle both enum and string inputs
            if isinstance(db_type, DatabaseType):
                filename = db_type.value
            else:
                filename = db_type

            # Check cache
            if use_cache and filename in self._cache:
                logger.info(f"Loading {filename} from cache")
                return LoadResult(
                    success=True,
                    data=self._cache[filename]['data'],
                    error=None,
                    metadata=self._cache[filename]['metadata']
                )

            # Construct file path
            file_path = self.base_path / filename

            if not file_path.exists():
                error_msg = f"Database file not found: {file_path}"
                logger.error(error_msg)
                return LoadResult(
                    success=False,
                    data=None,
                    error=error_msg,
                    metadata=None
                )

            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract metadata if available
            metadata = self._extract_metadata(data)

            # Validate structure
            validation_result = self._validate_structure(filename, data)
            if not validation_result['valid']:
                return LoadResult(
                    success=False,
                    data=None,
                    error=validation_result['error'],
                    metadata=metadata
                )

            # Cache the data
            self._cache[filename] = {
                'data': data,
                'metadata': metadata
            }

            logger.info(f"Successfully loaded {filename}")
            return LoadResult(
                success=True,
                data=data,
                error=None,
                metadata=metadata
            )

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in {filename}: {str(e)}"
            logger.error(error_msg)
            return LoadResult(
                success=False,
                data=None,
                error=error_msg,
                metadata=None
            )
        except Exception as e:
            error_msg = f"Error loading {filename}: {str(e)}"
            logger.error(error_msg)
            return LoadResult(
                success=False,
                data=None,
                error=error_msg,
                metadata=None
            )

    def load_multiple(self, db_types: List[Union[DatabaseType, str]],
                     use_cache: bool = True) -> Dict[str, LoadResult]:
        """
        Load multiple databases at once

        Args:
            db_types: List of database types to load
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping database names to LoadResult objects
        """
        results = {}
        for db_type in db_types:
            if isinstance(db_type, DatabaseType):
                key = db_type.value
            else:
                key = db_type
            results[key] = self.load_database(db_type, use_cache)
        return results

    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from loaded data"""
        metadata = {}

        # Common metadata fields
        metadata_fields = ['version', 'description', 'created_date',
                          'last_updated', 'total_entries', 'metadata']

        for field in metadata_fields:
            if field in data:
                metadata[field] = data[field]

        # Count entries if mappings or notes exist
        if 'mappings' in data:
            metadata['entry_count'] = len(data['mappings'])
        elif 'notes' in data:
            metadata['entry_count'] = len(data['notes'])

        return metadata

    def _validate_structure(self, filename: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure of loaded data"""
        validation = {'valid': True, 'error': None}

        # Specific validation for music_to_fragrance
        if 'music_to_fragrance' in filename:
            if 'mappings' not in data:
                validation = {
                    'valid': False,
                    'error': 'Missing required field: mappings'
                }
            elif not isinstance(data['mappings'], list):
                validation = {
                    'valid': False,
                    'error': 'mappings must be a list'
                }
            elif len(data['mappings']) > 0:
                # Check structure of first mapping
                sample = data['mappings'][0]
                required_fields = ['id', 'music', 'fragrance']
                for field in required_fields:
                    if field not in sample:
                        validation = {
                            'valid': False,
                            'error': f'Mapping missing required field: {field}'
                        }
                        break

        # Validation for texture_to_fragrance
        elif 'texture_to_fragrance' in filename:
            if 'mappings' not in data:
                validation = {
                    'valid': False,
                    'error': 'Missing required field: mappings'
                }
            elif len(data['mappings']) > 0:
                sample = data['mappings'][0]
                required_fields = ['id', 'texture', 'fragrance']
                for field in required_fields:
                    if field not in sample:
                        validation = {
                            'valid': False,
                            'error': f'Mapping missing required field: {field}'
                        }
                        break

        # Validation for fragrance notes database
        elif 'fragrance_notes' in filename:
            if 'notes' not in data:
                validation = {
                    'valid': False,
                    'error': 'Missing required field: notes'
                }

        return validation

    def get_cached_databases(self) -> List[str]:
        """Get list of currently cached databases"""
        return list(self._cache.keys())

    def clear_cache(self, filename: Optional[str] = None):
        """Clear cache for specific file or all files"""
        if filename:
            if filename in self._cache:
                del self._cache[filename]
                logger.info(f"Cleared cache for {filename}")
        else:
            self._cache.clear()
            logger.info("Cleared all cached data")

    def search_in_database(self, db_type: Union[DatabaseType, str],
                          query: str, field: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for entries in a database

        Args:
            db_type: Database to search in
            query: Search query string
            field: Specific field to search in (None searches all text fields)

        Returns:
            List of matching entries
        """
        result = self.load_database(db_type)
        if not result.success:
            logger.error(f"Failed to load database for search: {result.error}")
            return []

        matches = []
        query_lower = query.lower()

        # Search in mappings (for music/texture databases)
        if 'mappings' in result.data:
            for entry in result.data['mappings']:
                if self._search_in_entry(entry, query_lower, field):
                    matches.append(entry)

        # Search in notes (for fragrance notes database)
        elif 'notes' in result.data:
            for note_name, note_data in result.data['notes'].items():
                if field:
                    if field in note_data and query_lower in str(note_data[field]).lower():
                        matches.append({**note_data, 'note_name': note_name})
                else:
                    if query_lower in note_name.lower() or \
                       self._search_in_entry(note_data, query_lower, None):
                        matches.append({**note_data, 'note_name': note_name})

        return matches

    def _search_in_entry(self, entry: Union[Dict, List, str],
                        query: str, field: Optional[str] = None) -> bool:
        """Recursively search for query in entry"""
        if field and field in entry:
            return query in str(entry[field]).lower()

        if isinstance(entry, dict):
            for value in entry.values():
                if self._search_in_entry(value, query, None):
                    return True
        elif isinstance(entry, list):
            for item in entry:
                if self._search_in_entry(item, query, None):
                    return True
        elif isinstance(entry, str):
            return query in entry.lower()

        return False

    def get_statistics(self, db_type: Union[DatabaseType, str]) -> Dict[str, Any]:
        """Get statistics about a database"""
        result = self.load_database(db_type)
        if not result.success:
            return {'error': result.error}

        stats = {
            'filename': db_type.value if isinstance(db_type, DatabaseType) else db_type,
            'loaded': True
        }

        # Add metadata
        if result.metadata:
            stats.update(result.metadata)

        # Calculate additional statistics based on database type
        if 'mappings' in result.data:
            mappings = result.data['mappings']
            stats['total_mappings'] = len(mappings)

            # For music database
            if 'music_to_fragrance' in stats['filename']:
                genres = set()
                moods = set()
                for mapping in mappings:
                    if 'music' in mapping:
                        if 'genre' in mapping['music']:
                            genres.add(mapping['music']['genre'])
                        if 'mood' in mapping['music']:
                            moods.update(mapping['music']['mood'])
                stats['unique_genres'] = len(genres)
                stats['unique_moods'] = len(moods)

            # For texture database
            elif 'texture_to_fragrance' in stats['filename']:
                categories = set()
                for mapping in mappings:
                    if 'texture' in mapping and 'category' in mapping['texture']:
                        categories.add(mapping['texture']['category'])
                stats['texture_categories'] = list(categories)

        elif 'notes' in result.data:
            stats['total_notes'] = len(result.data['notes'])
            families = set()
            for note_data in result.data['notes'].values():
                if 'family' in note_data:
                    families.add(note_data['family'])
            stats['fragrance_families'] = len(families)

        return stats


# Convenience functions
def load_music_database(base_path: Optional[str] = None) -> LoadResult:
    """Load music to fragrance database"""
    loader = DataLoader(base_path)
    return loader.load_database(DatabaseType.MUSIC_TO_FRAGRANCE)


def load_texture_database(base_path: Optional[str] = None) -> LoadResult:
    """Load texture to fragrance database"""
    loader = DataLoader(base_path)
    return loader.load_database(DatabaseType.TEXTURE_TO_FRAGRANCE)


def load_all_databases(base_path: Optional[str] = None) -> Dict[str, LoadResult]:
    """Load all available databases"""
    loader = DataLoader(base_path)
    # Load the main databases
    db_types = [
        DatabaseType.MUSIC_TO_FRAGRANCE,
        DatabaseType.TEXTURE_TO_FRAGRANCE,
        DatabaseType.FRAGRANCE_NOTES,
    ]

    # Add other databases if they exist
    for db_type in DatabaseType:
        if db_type not in db_types:
            file_path = loader.base_path / db_type.value
            if file_path.exists():
                db_types.append(db_type)

    return loader.load_multiple(db_types)


if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")

    loader = DataLoader()

    # Test loading music database
    print("\n1. Loading music_to_fragrance_database.json...")
    music_result = loader.load_database(DatabaseType.MUSIC_TO_FRAGRANCE)
    if music_result.success:
        print(f"   [SUCCESS] Loaded {music_result.metadata.get('entry_count', 0)} mappings")
        print(f"   Version: {music_result.metadata.get('version', 'N/A')}")
    else:
        print(f"   [ERROR] {music_result.error}")

    # Test loading texture database
    print("\n2. Loading texture_to_fragrance_database.json...")
    texture_result = loader.load_database(DatabaseType.TEXTURE_TO_FRAGRANCE)
    if texture_result.success:
        print(f"   [SUCCESS] Loaded {texture_result.metadata.get('entry_count', 0)} mappings")
        print(f"   Version: {texture_result.metadata.get('version', 'N/A')}")
    else:
        print(f"   [ERROR] {texture_result.error}")

    # Test search functionality
    print("\n3. Testing search functionality...")
    jazz_results = loader.search_in_database(DatabaseType.MUSIC_TO_FRAGRANCE, "jazz")
    print(f"   Found {len(jazz_results)} entries matching 'jazz'")

    silk_results = loader.search_in_database(DatabaseType.TEXTURE_TO_FRAGRANCE, "silk")
    print(f"   Found {len(silk_results)} entries matching 'silk'")

    # Test statistics
    print("\n4. Getting statistics...")
    music_stats = loader.get_statistics(DatabaseType.MUSIC_TO_FRAGRANCE)
    print(f"   Music database: {music_stats.get('total_mappings', 0)} mappings, "
          f"{music_stats.get('unique_genres', 0)} genres")

    texture_stats = loader.get_statistics(DatabaseType.TEXTURE_TO_FRAGRANCE)
    print(f"   Texture database: {texture_stats.get('total_mappings', 0)} mappings, "
          f"categories: {texture_stats.get('texture_categories', [])}")

    print("\n[COMPLETE] All tests completed!")