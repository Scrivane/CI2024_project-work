#   *        Giovanni Squillero's GP Toolbox
#  / \       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2   +      A no-nonsense GP in pure Python
#    / \
#  10   11   Distributed under MIT License

from copy import deepcopy

from .node import Node
from .random import gxgp_random
from .gp_dag import DagGP
from .utils import arity

def find_parent(root, target):
        for node in root.subtree:
            for i, child in enumerate(node.successors):
                if child is target:
                    return node, i
        return None, None

def xover_swap_subtree(tree1: Node, tree2: Node) -> Node:
    offspring = deepcopy(tree1)
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf]

    if not internal_nodes:
        # No crossover possible
        return offspring
    node = gxgp_random.choice(internal_nodes)
    successors = node.successors

    i = gxgp_random.randrange(len(successors))
    successors[i] = deepcopy(gxgp_random.choice(list(tree2.subtree)))
    node.successors = successors
    return offspring





def mutation_hoist(tree1: Node) -> Node:
    offspring = deepcopy(tree1)
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf and node is not offspring]
    successors = None
    if not internal_nodes:
        return offspring

    # Pick random internal node 
    node = gxgp_random.choice(internal_nodes)

    return node




def mutation_collapse(tree1: Node,gptree:DagGP) -> Node:
    offspring = deepcopy(tree1)
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf]
    successors = None
    if not internal_nodes:
        return offspring
    


    # Pick random internal node 
    target = gxgp_random.choice(internal_nodes)
    possibilities= gptree._variables + gptree._constants  
    new_node = deepcopy(gxgp_random.choice(possibilities))

    parent, idx = find_parent(offspring, target)
    if parent is not None:
        children = parent.successors
        children[idx] = new_node
        parent.successors = children

    return offspring




def mutation_delete_unary(tree1: Node) -> Node:
    offspring = deepcopy(tree1)
    internal_nodes_arity_1 = [node for node in offspring.subtree if not node.is_leaf and node.arity==1]
    if not internal_nodes_arity_1:
        return offspring
    


    # Pick random internal node 
    target = gxgp_random.choice(internal_nodes_arity_1)


    parent, idx = find_parent(offspring, target)

    if parent is not None:
        child = target.successors[0]  #it's unary so there is only the first 
        succs = parent.successors  # This is a copy of the list done in successors , i need later to replace it whole otherwise no update
        succs[idx] = child         
        parent.successors = succs 
    
    else:  # i'm the root and i return the child
        return deepcopy(target.successors[0])
    return offspring





def mutation_add_unary(tree1: Node, gptree: DagGP) -> Node:
    offspring = deepcopy(tree1)

    all_nodes = list(offspring.subtree)
    unary_ops = [op for op in gptree._operators if arity(op) == 1]

    if not unary_ops or not all_nodes:
        return offspring  

    target = gxgp_random.choice(all_nodes)
    op = gxgp_random.choice(unary_ops)
    up_node = Node(op, [target])


    if target is offspring:
        
        return up_node

    parent, idx = find_parent(offspring, target)
    if parent is not None:
        succs = parent.successors
        succs[idx] = up_node
        parent.successors = succs
    


    return offspring

def mutation_point(tree1: Node,gptree:DagGP) -> Node:  
    offspring = deepcopy(tree1)  # single deepcopy
    
    all_nodes = list(offspring.subtree)

    target = gxgp_random.choice(all_nodes)
 
    if target.is_leaf:

        possibilities=[e for e in (gptree._variables + gptree._constants) if str(e)!=str(target)]
        new_node = deepcopy(gxgp_random.choice(possibilities))  

    else:  
       
        possible_ops = [op for op in gptree._operators if arity(op) == target.arity and target.short_name!=op.__name__  ]  
        if len(possible_ops)>0:
            new_op = gxgp_random.choice(possible_ops)
    
        new_node = Node(new_op, target.successors)


    if target is offspring:

        return new_node

    parent, idx = find_parent(offspring, target)
    if parent is not None:
        children = parent.successors
        children[idx] = new_node
        parent.successors = children



    return new_node
    


commutative_func=["multiply","add","hypot","maximum"]

def mutation_permutations(tree1: Node) -> Node:  
    offspring = deepcopy(tree1)  # single deepcopy

    
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf and node.arity>=2 and node.short_name not in commutative_func]  # uses non cummutative functions only 
    if (len(internal_nodes)>=1):
    

        target = gxgp_random.choice(internal_nodes)
        successors = target.successors
        succ1=deepcopy(successors[0])
        succ2=deepcopy(successors[1])

        target.successors=[succ2,succ1]
    

    return offspring



