
from kernel_discovery import description
from anytree import Node
from gpflow.kernels.linears import Polynomial
from kernel_discovery.description.simplify import simplify
from kernel_discovery.kernel import Product, Linear, Constant, RBF

_NOUN_PHRASES = {
    "periodic": "Periodic function",
    "white" : "Uncorrelated noise",
    "squared_exponential": "Smooth function",
    "constant": "Constant",
    "linear": "Linear function",
    'polynomial': "A polynomial function (of degree `{degree}`)"
}

_NOUN_PRECEDENCE = {
    'periodic': 0,
    'white': 1,
    'squared_exponential': 2,
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
    
    node = simplify(node)
    return _describe(node)
    
def _describe(node: Node):
    
    if node.is_leaf:
        return _NOUN_PHRASES[node.full_name.lower()] + ';'
    
    if node.name is Product:
        children = list(node.children[:])
        linear_child_count = [child.name for child in children].count(Linear)
        if linear_child_count > 1:
            children = [child for child in children if child.name is not Linear]
            children.append(Node(Polynomial, full_name='Polynomial', degree=linear_child_count))
        
        kernels_by_precedence = sorted(children, key=lambda child: _NOUN_PRECEDENCE[child.full_name.lower()])
        noun, *post_modifiers = kernels_by_precedence
        
        if noun.name is not Polynomial:
            noun_phrase = _NOUN_PHRASES[noun.full_name.lower()]
        else:
            noun_phrase = _NOUN_PHRASES[noun.full_name.lower()].format(degree=noun.degree)
        
        post_modifier_phrases = []
        for post_modifier in post_modifiers:
            if post_modifier.name is Constant:
                continue
            if post_modifier.name is not Polynomial:
                post_modifier_phrases.append(_POST_MODIFIERS[post_modifier.full_name.lower()])
            else:
                post_modifier_phrases.append(_POST_MODIFIERS[post_modifier.full_name.lower()].format(degree=noun.degree))
        
        return " ".join([noun_phrase] + post_modifier_phrases) + ";"
    
    return "\n".join([_describe(child) for child in node.children])


if __name__ == "__main__":
    
    # unit test
    from kernel_discovery.description.transform import kernel_to_ast
    def test_describe():
        k = (Linear() + RBF()) * Linear()
        ast = kernel_to_ast(k)
        
        desc = describe(ast)
        print(desc)
        
    test_describe()