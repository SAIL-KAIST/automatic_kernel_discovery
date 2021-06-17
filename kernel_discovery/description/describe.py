
from anytree import Node

_NOUN_PHRASES = {
    "periodic": "Periodic function",
    "white" : "Uncorrelated noise",
    "rbf": "Smooth function",
    "constant": "Constant",
    "linear": "Linear function",
    'polynomial': "A polynomial function (of degree `{degree}`)"
}

_NOUN_PRECEDENCE = {
    'periodic': 0,
    'white': 1,
    'rbf': 2,
    'constant': 3,
    'linear': 4,
    'polynomial': 5
}

_POST_MODIFIERS = {
    'rbf': 'whose shape changes smoothly',
    'periodic': 'modulated by a periodic function',
    'linear': 'with lineary varying amplitude',
    'polynomial': 'with polynomially varying amplitude of degree `{degree}`'
}

def describe(node: Node) -> str:
    
    pass