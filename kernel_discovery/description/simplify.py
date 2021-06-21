from anytree import Node
from copy import deepcopy
from kernel_discovery.description.transform import kernel_to_ast, ast_to_kernel
from kernel_discovery.kernel import Periodic, Product, RBF, Sum, White, Linear, Polynomial

def simplify(node: Node):
    
    to_simplify = deepcopy(node)
    return replace_white_product(
        merge_rbfs(
            distribution(
                to_simplify
            )
        )
    )
    
    
def replace_white_product(node: Node):
    node = deepcopy(node)
    _replace_white_product(node)
    return node

def _replace_white_product(node: Node):
    
    if node.is_leaf:
        return
    
    if node.name is Product:
        white_children = [child for child in node.children if child.name is White]
        if white_children:
            nonstationary_childen = [child for child in node.children
                                    if child.name in [Linear, Polynomial]]
            new_kids = [white_children[0]] + nonstationary_childen
            if len(new_kids) == 1:
                if node.is_root:
                    node.name = new_kids[0].name
                    try:
                        node.full_name = new_kids[0].full_name
                    except AttributeError:
                        pass
                else:
                    new_kids[0].parent = node.parent
                    node.parent = None
                node.children = []
            else:
                node.children = new_kids
    
    for child in node.children:
        _replace_white_product(child)

def merge_rbfs(node: Node):
    
    node = deepcopy(node)
    _merge_rbfs(node)
    return node
    
def _merge_rbfs(node:Node):
    
    if node.is_leaf:
        return
    
    if node.name is Product:
        rbf_children = [child for child in node.children if child.name is RBF]
        other_children = [child for child in node.children if child.name is not RBF]
        
        new_kids = other_children + rbf_children[:1]
        if len(new_kids) == 1:
            if node.is_root:
                node.name = new_kids[0].name
                try:
                    node.full_name = new_kids[0].full_name
                except AttributeError:
                    pass
            else:
                new_kids[0].parent = node.parent
                node.parent = None
            node.children = []
        else:
            node.children = new_kids
    
    for child in node.children:
        _merge_rbfs(child)
                

def distribution(node: Node):
    
    node = deepcopy(node)
    _distribution(node)
    return kernel_to_ast(ast_to_kernel(node))

def _distribution(node:Node):
    
    if node.is_leaf:
        return
    
    if node.name is Product:
        sum_to_distribute = [child for child in node.children if child.name is Sum]
        
        if sum_to_distribute:
            sum_to_distr = sum_to_distribute[0]
            children_to_distribute_to = [child for child in node.children if child is not sum_to_distr]
            
            node.name = Sum
            node.full_name = 'Sum'
            node.children = []
            
            for child in sum_to_distr.children:
                new_prod = Node(name=Product, full_name='Product', parent=node)
                
                new_kids = [deepcopy(child) for child in children_to_distribute_to]
                if child.name is Product:
                    new_kids.extend([deepcopy(c) for c in child.children])
                else:
                    new_kids += [child]
                
                for kid in new_kids:
                    kid.parent = new_prod
        
    for child in node.children:
        _distribution(child)
        
        
# unit test
if __name__ == "__main__":
    
    from anytree import LevelOrderIter
    
    def are_asts_equal(ast1, ast2):
        
        for a, b in zip(LevelOrderIter(ast1), LevelOrderIter(ast2)):
            if a.name != b.name:
                return False
        return True
    
    def test_distribution():
        k = (RBF() + White() *Linear()) * Polynomial()
        ast = kernel_to_ast(k)
        
        # Level 1.
        ast_should_be = Node(Sum, full_name='Sum')

        # Level 2.
        p1 = Node(Product, full_name='Product', parent=ast_should_be)
        p2 = Node(Product, full_name='Product', parent=ast_should_be)

        # Level 3.1
        Node(Polynomial, parent=p1)
        Node(RBF, parent=p1)

        # Level 3.2
        Node(Polynomial, parent=p2)
        Node(White, parent=p2)
        Node(Linear, parent=p2)
        
        assert are_asts_equal(distribution(ast), ast_should_be)
        
    def test_merge_rbfs():
        
        k = ((RBF() * RBF() + RBF() + RBF() + White() * Linear())) * Polynomial() * RBF() * RBF()
        
        ast = kernel_to_ast(k)
        merged_rbf_ast = merge_rbfs(ast)        
                
        ast_should_be = Node(Product, full_name='Product')

        p1 = Node(Sum, full_name='Sum', parent=ast_should_be)
        Node(Polynomial, full_name='Polynomial', parent=ast_should_be)
        Node(RBF, full_name='RBF', parent=ast_should_be)

        Node(RBF, full_name='RBF', parent=p1)
        Node(RBF, full_name='RBF', parent=p1)
        p2 = Node(Product, full_name='Product', parent=p1)
        Node(RBF, full_name='RBF', parent=p1)

        Node(White, full_name='White', parent=p2)
        Node(Linear, full_name='Linear', parent=p2)
        
        assert are_asts_equal(merged_rbf_ast, ast_should_be)
        
    def test_replace_white_products():
        
        k1 = Product([White(), White(), White(), White()])
        ast1 = kernel_to_ast(k1)
        ast1_should_be = Node(White)
        assert are_asts_equal(ast1_should_be, replace_white_product(ast1))
        
        k2 = White() + White()
        ast2 = kernel_to_ast(k2)
        ast2_should_be = Node(Sum)
        Node(White, parent=ast2_should_be)
        Node(White, parent=ast2_should_be)
        assert are_asts_equal(ast2_should_be, replace_white_product(ast2))
        
        k3 = Product([White(), White(), White(), White(), RBF(), Linear(), Polynomial()]) + RBF()
        
        ast3 =kernel_to_ast(k3)
        ast3_should_be = Node(Sum)
        p1 = Node(Product, parent=ast3_should_be)
        Node(RBF, parent=ast3_should_be)
        
        Node(White, parent=p1)
        Node(Linear, parent=p1)
        Node(Polynomial, parent=p1)
        
        assert are_asts_equal(ast3_should_be, replace_white_product(ast3))
        
        k = RBF() * White() + Periodic() * White()
        
        ast = kernel_to_ast(k)
        
        ast_should_be = Node(Sum, full_name='Sum')
        Node(White, full_name='White', parent=ast_should_be)
        Node(White, full_name='White', parent=ast_should_be)
        
        assert are_asts_equal(ast_should_be, replace_white_product(ast))
    
    def test_simplify():
        
        k = (Linear() + RBF()) * Periodic() * RBF() + White() * Linear() * Periodic() * White()
        
        ast = kernel_to_ast(k)
        
        ast_should_be = Node(Sum, full_name='Sum')
        
        prod1 = Node(Product, full_name='Product', parent=ast_should_be)
        Node(Periodic, full_name='Periodic', parent=prod1)
        Node(Linear, full_name='Linear', parent=prod1)
        Node(RBF, full_name='RBF', parent=prod1)
        
        prod2 = Node(Product, full_name='Product', parent=ast_should_be)
        Node(Periodic, full_name='Periodic', parent=prod2)
        Node(RBF, full_name='RBF', parent=prod2)
    
        prod3 = Node(Product, full_name='Product', parent=ast_should_be)
        Node(White, full_name='White', parent=prod3)
        Node(Linear, full_name='Linear', parent=prod3)
        
        assert are_asts_equal(ast_should_be, simplify(ast))
        assert not are_asts_equal(ast, simplify(ast))
    
    # test_distribution()
    
    # test_merge_rbfs()
    
    # test_replace_white_products()
    
    test_simplify()
    