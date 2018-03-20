import networkx as nx
import gridtools.intermediate_representation.graph as irg
import gridtools.intermediate_representation.utils as irutils


def translate_graph(graph):
    # Populate the list of pending nodes with root nodes of the graph
    pending = irutils.find_roots(graph)

    stage_src = list()

    while pending:
        node = pending.pop()

        # If pending node is not a leaf, this is the start of a new statement
        # in the code
        if graph.successors(node):
            stage_src.append(start_dfs_translate(node, graph, pending))

    # The DFS traversals have generated the source lines from the last to the
    # first. To have an executable source code, we need to reverse the list
    stage_src.reverse()

    return stage_src


def start_dfs_translate(root, graph, pending):
    indices = ("i", "j", "k")[:root.rank]
    child = irutils.get_successor(graph, root)
    edge = irutils.get_out_edge(graph, root)

    expr = "{}[{}] = ".format(str(root), ",".join(indices))
    expr += dfs_translate(child, graph, pending, edge)

    return expr


def dfs_translate(node, graph, pending, edge):

    if type(node) is irg.NodeNamedExpression:
        pending.append(node)

        access_offsets = list(edge.indices_offsets)
        indexing_string = generate_indexing_string_for_named_expression(access_offsets)

        return "{0}{1}".format(str(node), indexing_string)

    if type(node) is irg.NodeConstant:
        return str(node)

    if type(node) is irg.NodeBinaryOperator:
        left_node = irutils.get_successor_left(graph, node)
        right_node = irutils.get_successor_right(graph, node)

        # Propagating node data is a clean way to correctly process the case in
        # which multiple binary operations point to the same named expression,
        # e.g. a Laplacian operator
        left_edge = irutils.get_out_edge_left(graph, node)
        right_edge = irutils.get_out_edge_right(graph, node)

        left = dfs_translate(left_node, graph, pending, edge=left_edge)
        right = dfs_translate(right_node, graph, pending, edge=right_edge)

        return "({0}) {1} ({2})".format(left, str(node), right)


def generate_indexing_string_for_named_expression(access_offsets):
    # Ensure the offset list has 3 elements
    if len(access_offsets) > 3:
        raise ValueError("Python debug backend for now only supports array "
                         "offsets in 3 dimensions!")

    indices = ("i", "j", "k")[:len(access_offsets)]

    offsets_strings = []
    for ind, off in zip(indices, access_offsets):
        if off == 0:
            offsets_strings.append(ind)
        elif off > 0:
            offsets_strings.append("{}+{}".format(ind, str(off)))
        else:
            offsets_strings.append("{}{}".format(ind, str(off)))

    return "[{}]".format(",".join(offsets_strings))
