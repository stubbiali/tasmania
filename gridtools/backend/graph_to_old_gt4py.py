import networkx as nx
import gridtools.intermediate_representation.graph as irg
import gridtools.intermediate_representation.utils as irutils


def translate_graph(graph):
    # Populate the list of pending nodes with root nodes of the graph
    pending = irutils.find_roots(graph)

    # The roots of the graph are also stage outputs and stage arguments
    stage_outputs = [str(out_node) for out_node in pending]
    stage_args = [str(out_node) for out_node in pending]

    stage_src = list()

    while pending:
        node = pending.pop()

        # Check if pending node is a leaf
        if not graph.successors(node):
            # Add its symbol to the stage arguments, if not already registered
            new_arg = str(node)
            if new_arg not in stage_args:
                stage_args.append(new_arg)
        else:
            # This is the start of a new statement in the code
            stage_src.append(start_dfs_translate(node, graph, pending))

    # The DFS traversals have generated the source lines from the last to the
    # first. To have an executable source code, we need to reverse the list
    stage_src.reverse()

    return stage_src, stage_args, stage_outputs


def start_dfs_translate(root, graph, pending):
    child = irutils.get_successor(graph, root)
    edge = irutils.get_out_edge(graph, root)

    expr = "{0}[p] = ".format(str(root))
    expr += dfs_translate(child, graph, pending, edge)

    return expr


def dfs_translate(node, graph, pending, edge):

    if type(node) is irg.NodeNamedExpression:
        pending.append(node)

        access_offsets = list(edge.indices_offsets)
        indexing_string = generate_old_indexing_string_for_named_expression(access_offsets)

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


def generate_old_indexing_string_for_named_expression(access_offsets):
    # Ensure the offset list has 3 elements
    if len(access_offsets) > 3:
        raise ValueError("Old GT4Py API only supports array offsets in 3 dimensions!")
    while len(access_offsets) < 3:
        access_offsets.append(0)

    # Generate the offset string with a syntax suitable to be inserted in the
    # indexing expression
    if all(off == 0 for off in access_offsets):
        offset_str = ""
    else:
        offset_str = " + ({0})".format(",".join([str(off) for off in access_offsets]))

    return "[p{0}]".format(offset_str)
