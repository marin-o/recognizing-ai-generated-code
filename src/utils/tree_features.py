"""
Tree-sitter feature extraction utility for multiple programming languages.
Supports: Python, C, C++, C#, Java, JavaScript, PHP, and Go.

This module extracts both AST structural features and statistical features 
from source code for use in machine learning models.
"""

import logging
from typing import Dict, List, Tuple, Optional
from tree_sitter import Language, Parser, Node
import numpy as np

# Import language parsers
try:
    import tree_sitter_python as tspython
    HAS_PYTHON = True
except ImportError:
    HAS_PYTHON = False

try:
    import tree_sitter_cpp as tscpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

try:
    import tree_sitter_java as tsjava
    HAS_JAVA = True
except ImportError:
    HAS_JAVA = False

try:
    import tree_sitter_c_sharp as tscsharp
    HAS_CSHARP = True
except ImportError:
    HAS_CSHARP = False

try:
    import tree_sitter_javascript as tsjavascript
    HAS_JAVASCRIPT = True
except ImportError:
    HAS_JAVASCRIPT = False

try:
    import tree_sitter_php as tsphp
    HAS_PHP = True
except ImportError:
    HAS_PHP = False

try:
    import tree_sitter_go as tsgo
    HAS_GO = True
except ImportError:
    HAS_GO = False

# Setup logging
logger = logging.getLogger(__name__)

# Language-specific node types for nesting depth calculation
NESTING_NODES = {
    'python': {
        "function_definition", "class_definition", "if_statement", 
        "for_statement", "while_statement", "with_statement", "try_statement"
    },
    'c': {
        "function_definition", "if_statement", "for_statement", 
        "while_statement", "switch_statement", "do_statement"
    },
    'cpp': {
        "function_definition", "class_specifier", "struct_specifier", 
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "namespace_definition", "try_statement", "do_statement"
    },
    'c++': {  # Alias for cpp
        "function_definition", "class_specifier", "struct_specifier", 
        "if_statement", "for_statement", "while_statement", "switch_statement",
        "namespace_definition", "try_statement", "do_statement"
    },
    'csharp': {
        "method_declaration", "constructor_declaration", "class_declaration",
        "struct_declaration", "if_statement", "for_statement", "foreach_statement",
        "while_statement", "switch_statement", "try_statement", "do_statement"
    },
    'java': {
        "method_declaration", "constructor_declaration", "class_declaration", 
        "interface_declaration", "if_statement", "for_statement", "while_statement", 
        "enhanced_for_statement", "switch_statement", "try_statement", "do_statement"
    },
    'javascript': {
        "function_declaration", "function", "method_definition", "class_declaration",
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "switch_statement", "try_statement", "do_statement"
    },
    'php': {
        "function_definition", "method_declaration", "class_declaration",
        "if_statement", "for_statement", "foreach_statement", "while_statement",
        "switch_statement", "try_statement", "do_statement"
    },
    'go': {
        "function_declaration", "method_declaration", "if_statement",
        "for_statement", "switch_statement", "type_switch_statement", "select_statement"
    }
}

# Language-specific queries for feature extraction
LANGUAGE_QUERIES = {
    'python': {
        'functions': "(function_definition) @func",
        'classes': "(class_definition) @class_def",
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt"],
        'imports': ["(import_statement) @import", "(import_from_statement) @import_from"],
        'comments': "(comment) @comment",
        'binary_ops': "(binary_operator) @binop",
        'errors': "(ERROR) @error"
    },
    'c': {
        'functions': "(function_definition) @func",
        'classes': "(struct_specifier) @struct_def",
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt", "(do_statement) @do_stmt"],
        'imports': "(preproc_include) @include",
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'cpp': {
        'functions': "(function_definition) @func",
        'classes': ["(class_specifier) @class_def", "(struct_specifier) @struct_def"],
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt", "(do_statement) @do_stmt"],
        'imports': ["(preproc_include) @include", "(using_declaration) @using"],
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'c++': {  # Alias for cpp
        'functions': "(function_definition) @func",
        'classes': ["(class_specifier) @class_def", "(struct_specifier) @struct_def"],
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt", "(do_statement) @do_stmt"],
        'imports': ["(preproc_include) @include", "(using_declaration) @using"],
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'csharp': {
        'functions': ["(method_declaration) @method", "(constructor_declaration) @constructor"],
        'classes': ["(class_declaration) @class_def", "(struct_declaration) @struct_def"],
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(foreach_statement) @foreach_stmt", "(while_statement) @while_stmt"],
        'imports': "(using_directive) @using",
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'java': {
        'functions': ["(method_declaration) @method", "(constructor_declaration) @constructor"],
        'classes': ["(class_declaration) @class_def", "(interface_declaration) @interface_def"],
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(while_statement) @while_stmt", "(enhanced_for_statement) @enhanced_for"],
        'imports': "(import_declaration) @import",
        'comments': ["(line_comment) @line_comment", "(block_comment) @block_comment"],
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'javascript': {
        'functions': ["(function_declaration) @func", "(function) @func_expr", "(arrow_function) @arrow_func", "(method_definition) @method"],
        'classes': "(class_declaration) @class_def",
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(for_in_statement) @for_in", "(while_statement) @while_stmt", "(do_statement) @do_stmt"],
        'imports': ["(import_statement) @import"],
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'php': {
        'functions': ["(function_definition) @func", "(method_declaration) @method"],
        'classes': "(class_declaration) @class_def",
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt", "(foreach_statement) @foreach_stmt", "(while_statement) @while_stmt"],
        'imports': ["(namespace_use_declaration) @use"],
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    },
    'go': {
        'functions': ["(function_declaration) @func", "(method_declaration) @method"],
        'classes': "(type_declaration) @type_decl",
        'if_statements': "(if_statement) @if_stmt",
        'loops': ["(for_statement) @for_stmt"],
        'imports': "(import_declaration) @import",
        'comments': "(comment) @comment",
        'binary_ops': "(binary_expression) @binop",
        'errors': "(ERROR) @error"
    }
}


def get_language_parser(language: str) -> Tuple[Language, Parser]:
    """
    Get the appropriate Tree-sitter language and parser for a given language.
    
    Args:
        language: Programming language name (case-insensitive)
        
    Returns:
        Tuple of (Language, Parser)
        
    Raises:
        ValueError: If language is not supported or parser not available
    """
    lang_lower = language.lower().replace('#', 'sharp')  # Handle C# -> csharp
    
    if lang_lower == 'python':
        if not HAS_PYTHON:
            raise ValueError("tree_sitter_python not installed")
        lang = Language(tspython.language())
    elif lang_lower in ['cpp', 'c++']:
        if not HAS_CPP:
            raise ValueError("tree_sitter_cpp not installed")
        lang = Language(tscpp.language())
    elif lang_lower == 'c':
        if not HAS_CPP:  # C parser uses cpp library
            raise ValueError("tree_sitter_cpp not installed (needed for C)")
        lang = Language(tscpp.language())
    elif lang_lower == 'csharp':
        if not HAS_CSHARP:
            raise ValueError("tree_sitter_c_sharp not installed")
        lang = Language(tscsharp.language())
    elif lang_lower == 'java':
        if not HAS_JAVA:
            raise ValueError("tree_sitter_java not installed")
        lang = Language(tsjava.language())
    elif lang_lower in ['javascript', 'js']:
        if not HAS_JAVASCRIPT:
            raise ValueError("tree_sitter_javascript not installed")
        lang = Language(tsjavascript.language())
    elif lang_lower == 'php':
        if not HAS_PHP:
            raise ValueError("tree_sitter_php not installed")
        lang = Language(tsphp.language_php())
    elif lang_lower == 'go':
        if not HAS_GO:
            raise ValueError("tree_sitter_go not installed")
        lang = Language(tsgo.language())
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    parser = Parser(lang)
    return lang, parser


def traverse_for_depth(node: Node, depth: int, language: str) -> int:
    """
    Traverse the AST and compute the maximum nesting depth.
    
    Args:
        node: Current node in the AST
        depth: Current depth in the tree
        language: Programming language
        
    Returns:
        Maximum nesting depth encountered
    """
    max_nesting_depth = depth
    
    node_type = node.type
    nesting_types = NESTING_NODES.get(language.lower().replace('#', 'sharp'), set())
    
    if node_type in nesting_types:
        for child in node.children:
            max_nesting_depth = max(max_nesting_depth, traverse_for_depth(child, depth + 1, language))
    else:
        for child in node.children:
            max_nesting_depth = max(max_nesting_depth, traverse_for_depth(child, depth, language))
    
    return max_nesting_depth


def count_nodes(node: Node) -> int:
    """
    Count total number of nodes in AST.
    
    Args:
        node: Root node of AST
        
    Returns:
        Total node count
    """
    count = 1
    for child in node.children:
        count += count_nodes(child)
    return count


def count_nodes_by_type(node: Node, node_types: set) -> int:
    """
    Count nodes of specific types by traversing the tree.
    
    Args:
        node: Root node to start traversal
        node_types: Set of node type strings to count
        
    Returns:
        Total count of matching nodes
    """
    count = 1 if node.type in node_types else 0
    for child in node.children:
        count += count_nodes_by_type(child, node_types)
    return count


def execute_queries(lang: Language, root_node: Node, queries: List[str]) -> int:
    """
    Execute multiple node type searches and return total count.
    
    Args:
        lang: Tree-sitter Language object (unused, kept for compatibility)
        root_node: Root node of AST
        queries: List of node type strings (e.g., "function_definition")
        
    Returns:
        Total count of matches across all node types
    """
    # Extract node types from query strings
    # Query format: "(node_type) @name" -> extract "node_type"
    node_types = set()
    for query_str in queries:
        try:
            # Extract node type from query string like "(function_definition) @func"
            if '(' in query_str and ')' in query_str:
                node_type = query_str.split('(')[1].split(')')[0].strip()
                node_types.add(node_type)
        except Exception as e:
            logger.debug(f"Failed to parse query: {query_str}, error: {e}")
            continue
    
    return count_nodes_by_type(root_node, node_types)


def safe_execute_single_query(lang: Language, root_node: Node, query_str: str) -> int:
    """
    Execute a single node type search safely.
    
    Args:
        lang: Tree-sitter Language object (unused, kept for compatibility)
        root_node: Root node of AST
        query_str: Query string (e.g., "(function_definition) @func")
        
    Returns:
        Count of matches, or 0 if query fails
    """
    try:
        # Extract node type from query string
        if '(' in query_str and ')' in query_str:
            node_type = query_str.split('(')[1].split(')')[0].strip()
            return count_nodes_by_type(root_node, {node_type})
        return 0
    except Exception as e:
        logger.debug(f"Query failed: {query_str}, error: {e}")
        return 0


def extract_tree_features(code: str, language: str) -> Dict[str, float]:
    """
    Extract comprehensive AST and statistical features from source code.
    
    Features extracted:
    - Structural: function_count, class_count, if_count, loop_count, import_count
    - Code quality: comment_count, binary_op_count, error_count
    - Complexity: max_nesting_depth, total_nodes, avg_node_depth
    
    Args:
        code: Source code string
        language: Programming language name
        
    Returns:
        Dictionary with feature names and values. Returns default values if parsing fails.
    """
    # Default features for empty or failed cases
    default_features = {
        'function_count': 0.0,
        'class_count': 0.0,
        'if_count': 0.0,
        'loop_count': 0.0,
        'import_count': 0.0,
        'comment_count': 0.0,
        'binary_op_count': 0.0,
        'error_count': 1.0,  # 1.0 indicates parse error
        'max_nesting_depth': 0.0,
        'total_nodes': 0.0,
        'avg_node_depth': 0.0,
    }
    
    if not code or not code.strip():
        logger.debug("Empty code provided")
        return default_features
    
    try:
        # Get appropriate parser
        lang, parser = get_language_parser(language)
        
        # Parse code
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        
        # Get language-specific queries
        lang_lower = language.lower().replace('#', 'sharp')
        if lang_lower not in LANGUAGE_QUERIES:
            # Try to use a similar language's queries as fallback
            if lang_lower in ['c++', 'cpp']:
                lang_lower = 'cpp'
            else:
                logger.warning(f"No queries defined for language: {language}, using defaults")
                return default_features
        
        queries = LANGUAGE_QUERIES[lang_lower]
        
        # Extract features
        features = {}
        
        # Functions
        if isinstance(queries['functions'], list):
            features['function_count'] = float(execute_queries(lang, root_node, queries['functions']))
        else:
            features['function_count'] = float(safe_execute_single_query(lang, root_node, queries['functions']))
        
        # Classes
        if isinstance(queries['classes'], list):
            features['class_count'] = float(execute_queries(lang, root_node, queries['classes']))
        else:
            features['class_count'] = float(safe_execute_single_query(lang, root_node, queries['classes']))
        
        # If statements
        features['if_count'] = float(safe_execute_single_query(lang, root_node, queries['if_statements']))
        
        # Loops
        features['loop_count'] = float(execute_queries(lang, root_node, queries['loops']))
        
        # Imports
        if isinstance(queries['imports'], list):
            features['import_count'] = float(execute_queries(lang, root_node, queries['imports']))
        else:
            features['import_count'] = float(safe_execute_single_query(lang, root_node, queries['imports']))
        
        # Comments
        if isinstance(queries['comments'], list):
            features['comment_count'] = float(execute_queries(lang, root_node, queries['comments']))
        else:
            features['comment_count'] = float(safe_execute_single_query(lang, root_node, queries['comments']))
        
        # Binary operations
        features['binary_op_count'] = float(safe_execute_single_query(lang, root_node, queries['binary_ops']))
        
        # Errors
        features['error_count'] = float(safe_execute_single_query(lang, root_node, queries['errors']))
        
        # Nesting depth
        features['max_nesting_depth'] = float(traverse_for_depth(root_node, 0, language))
        
        # Total nodes
        features['total_nodes'] = float(count_nodes(root_node))
        
        # Average node depth (approximate using total nodes and max depth)
        if features['max_nesting_depth'] > 0:
            features['avg_node_depth'] = features['total_nodes'] / (features['max_nesting_depth'] + 1)
        else:
            features['avg_node_depth'] = features['total_nodes']
        
        return features
        
    except ValueError as e:
        # Language not supported or parser not available
        logger.warning(f"Language parser error for {language}: {str(e)}")
        return default_features
    except Exception as e:
        logger.warning(f"Failed to extract tree features for {language}: {str(e)}")
        return default_features


def get_feature_vector(code: str, language: str) -> np.ndarray:
    """
    Get feature vector as numpy array for easy integration with ML models.
    
    Args:
        code: Source code string
        language: Programming language name
        
    Returns:
        Numpy array of shape (11,) containing features in a fixed order
    """
    features = extract_tree_features(code, language)
    
    # Fixed order for consistency
    feature_order = [
        'function_count', 'class_count', 'if_count', 'loop_count',
        'import_count', 'comment_count', 'binary_op_count', 'error_count',
        'max_nesting_depth', 'total_nodes', 'avg_node_depth'
    ]
    
    return np.array([features[k] for k in feature_order], dtype=np.float32)


def get_supported_languages() -> List[str]:
    """
    Get list of supported languages based on installed parsers.
    
    Returns:
        List of language names that can be parsed
    """
    supported = []
    
    if HAS_PYTHON:
        supported.append('python')
    if HAS_CPP:
        supported.extend(['c', 'cpp', 'c++'])
    if HAS_JAVA:
        supported.append('java')
    if HAS_CSHARP:
        supported.append('csharp')
    if HAS_JAVASCRIPT:
        supported.extend(['javascript', 'js'])
    if HAS_PHP:
        supported.append('php')
    if HAS_GO:
        supported.append('go')
    
    return supported


def get_feature_dimension() -> int:
    """
    Get the dimension of the feature vector.
    
    Returns:
        Integer dimension (always 11 for current implementation)
    """
    return 11


if __name__ == "__main__":
    # Test the feature extraction
    logging.basicConfig(level=logging.INFO)
    
    print("Supported languages:", get_supported_languages())
    print("Feature dimension:", get_feature_dimension())
    print()
    
    # Test with Python code
    python_code = """
def calculate_sum(numbers):
    '''Calculate sum of numbers'''
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    return total

class Calculator:
    def add(self, a, b):
        return a + b
"""
    
    print("Testing Python code:")
    features = extract_tree_features(python_code, "python")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\nFeature vector shape:", get_feature_vector(python_code, "python").shape)
    print("Feature vector:", get_feature_vector(python_code, "python"))
    
    # Test with Java code
    java_code = """
public class Example {
    public static void main(String[] args) {
        for (int i = 0; i < 10; i++) {
            if (i % 2 == 0) {
                System.out.println(i);
            }
        }
    }
}
"""
    
    print("\n\nTesting Java code:")
    features = extract_tree_features(java_code, "java")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Test with unsupported/empty code
    print("\n\nTesting empty code:")
    features = extract_tree_features("", "python")
    print("Empty code features:", features)
