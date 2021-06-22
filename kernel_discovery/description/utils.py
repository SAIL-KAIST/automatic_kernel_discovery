from anytree import AsciiStyle, Node, RenderTree

def pretty_ast(ast: Node) -> str:
    
    try:
        ast.full_name
        return RenderTree(ast, style=AsciiStyle()).by_attr('full_name')
    except AttributeError:
        return RenderTree(ast, style=AsciiStyle()).by_attr('name')