import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plot_node(ax, node_text, center_pt, parent_pt, node_type):
    ax.annotate(node_text, 
                xy=parent_pt, 
                xycoords='axes fraction',
                xytext=center_pt, 
                textcoords='axes fraction',
                va='center',
                ha='center',
                bbox=node_type,
                arrowprops=arrow_args)


def get_num_leafs(tree: dict):
    num_leaf = 0

    feature_label = list(tree)[0]
    feature_values = list(tree[feature_label])

    for feature_value in feature_values:
        if type(tree[feature_label][feature_value]) is dict:
            num_leaf += get_num_leafs(tree[feature_label][feature_value])
        else:
            num_leaf += 1

    return num_leaf


def get_tree_depth(tree: dict):
    depth = 1

    feature_label = list(tree)[0]
    feature_values = list(tree[feature_label])

    for feature_value in feature_values:
        if type(tree[feature_label][feature_value]) is dict:
            new_depth = get_tree_depth(tree[feature_label][feature_value]) + 1
            if new_depth > depth:
                depth = new_depth

    return depth


def plot_mid_text(ax, cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] + cntr_pt[0]) / 2
    y_mid = (parent_pt[1] + cntr_pt[1]) / 2
    ax.text(x_mid, y_mid, txt_string, horizontalalignment='center', 
            rotation=45)


def plot_tree(ax, tree, x_off, y_off, total_w, total_d, parent_pt, node_text):
    """Visualise the decision tree

    params
    ======
    ax: matplotlib ax
    tree: the decision tree to visualise
    x_off: x position of the most left leaf in the tree
    y_off: y position of the parent of the tree
    total_w: the total number of leafs of the original tree
    total_d: the depth of the origianl tree
    parent_pt: the position of the parent node for the tree
    node_text: the value split the sub-dataset from the original dataset
    """

    num_leafs = get_num_leafs(tree)
    feature_label = list(tree)[0]

    cntr_pt = (x_off + (1. + num_leafs)/2./total_w, y_off)
    plot_mid_text(ax, cntr_pt, parent_pt, node_text)
    plot_node(ax, feature_label, cntr_pt, parent_pt, decision_node)
    
    feature_values = list(tree[feature_label])

    y_off = y_off - 1./total_d
    for feature_value in feature_values:
        if type(tree[feature_label][feature_value]) is dict:
            sub_tree = tree[feature_label][feature_value]
            sub_tree_w = get_num_leafs(sub_tree)
            x_off += sub_tree_w/total_w
            plot_tree(ax, sub_tree, x_off - sub_tree_w/total_w, y_off, 
                      total_w, total_d, cntr_pt, str(feature_value))
        else:
            x_off += 1.0/total_w
            plot_node(ax, tree[feature_label][feature_value], (x_off, y_off), 
                      cntr_pt, leaf_node)
            plot_mid_text(ax, (x_off, y_off), cntr_pt, str(feature_value))


def create_plot(input_tree):
    """Start the creation of a tree from this function
    """
    
    fig = plt.figure(facecolor='white')
    fig.clf()
    ax_props = dict(xticks=[], yticks=[])

    ax1 = plt.subplot(111, frameon=False, **ax_props)
    total_w = get_num_leafs(input_tree)
    total_d = get_tree_depth(input_tree)
    x_off = -0.5/total_w
    y_off = 1.
    plot_tree(ax1, input_tree, x_off, y_off, total_w, total_d, (0.5, 1.0), '')
    plt.show()
    

def retreve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 
                                       1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 
                                       1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 
                                                        1: 'no'}}}}]
    return list_of_trees[i]


if __name__ == '__main__':
    my_tree = retreve_tree(0)
    my_tree['no surfacing'][3] = 'maybe'
    create_plot(my_tree)