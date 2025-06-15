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
    #ic(offspring)
    node = gxgp_random.choice(internal_nodes)
    successors = node.successors

    i = gxgp_random.randrange(len(successors))
    #ic(successors, i)
    successors[i] = deepcopy(gxgp_random.choice(list(tree2.subtree)))
    #ic(successors, i)
    node.successors = successors
    return offspring





def mutation_hoist(tree1: Node) -> Node:
    offspring = deepcopy(tree1)
    #ic(offspring)
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf and node is not offspring]
    successors = None
    if not internal_nodes:
        return offspring

    # Pick random internal node (i.e., a subtree)
    node = gxgp_random.choice(internal_nodes)
    """ i = gxgp_random.randrange(len(successors))
    ic(successors, i)
    successors[i] = deepcopy(gxgp_random.choice(list(tree2.subtree)))
    ic(successors, i)
    node.successors = successors """
    #ic(node)
    return node




def mutation_collapse(tree1: Node,gptree:DagGP) -> Node:
    offspring = deepcopy(tree1)
    #ic(offspring)
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf]
    successors = None
    if not internal_nodes:
        return offspring
    


    # Pick random internal node (i.e., a subtree)
    target = gxgp_random.choice(internal_nodes)
    possibilities= gptree._variables + gptree._constants  #[e for e in (gptree._variables + gptree._constants) if str(e)!=str(target)]
    new_node = deepcopy(gxgp_random.choice(possibilities))

    parent, idx = find_parent(offspring, target)
    if parent is not None:
        children = parent.successors
        children[idx] = new_node
        parent.successors = children

    return offspring




def mutation_delete_unirary(tree1: Node) -> Node:
    offspring = deepcopy(tree1)
    #ic(offspring)
    internal_nodes_arity_1 = [node for node in offspring.subtree if not node.is_leaf and node.arity==1]
    if not internal_nodes_arity_1:
        return offspring
    


    # Pick random internal node (i.e., a subtree)
    target = gxgp_random.choice(internal_nodes_arity_1)
    #possibilities= gptree._variables + gptree._constants  #[e for e in (gptree._variables + gptree._constants) if str(e)!=str(target)]
    #new_node = deepcopy(gxgp_random.choice(possibilities))

    parent, idx = find_parent(offspring, target)

    if parent is not None:
        child = target.successors[0]  #it's unirary so there is only the first 
        #children[idx] = new_node
        succs = parent.successors  # This is a copy of the list done in successors , i need later to replace it whole otherwise no update
        succs[idx] = child         
        parent.successors = succs 
    
    else:  # i'm the root and i return the child
        return deepcopy(target.successors[0])
    return offspring





def mutation_add_unirary(tree1: Node, gptree: DagGP) -> Node:
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

def mutation_point(tree1: Node,gptree:DagGP) -> Node:  ##i have the problem that chainging a node changes all similar nodes
    offspring = deepcopy(tree1)  # single deepcopy
    
    all_nodes = list(offspring.subtree)

    target = gxgp_random.choice(all_nodes)
 
    if target.is_leaf:

        possibilities=[e for e in (gptree._variables + gptree._constants) if str(e)!=str(target)]
        new_node = deepcopy(gxgp_random.choice(possibilities))  

    else:  #make so only changes
       
        possible_ops = [op for op in gptree._operators if arity(op) == target.arity and target.short_name!=op.__name__  ]  #con reimbussolamento
        if len(possible_ops)>0:
            new_op = gxgp_random.choice(possible_ops)
        else:
            "qui avrei errore"
        new_node = Node(new_op, target.successors)




    

    if target is offspring:

        return new_node

    parent, idx = find_parent(offspring, target)
    if parent is not None:
        children = parent.successors
        children[idx] = new_node
        parent.successors = children

    # Replace target in its parent
    """ for node in all_nodes:
        children = node.successors
        for i, child in enumerate(children):
            if child is target:  # identity check
                children[i] = new_node
                node.successors = children  # this triggers the setter
                print(offspring)
                return offspring """


    return new_node
    


commutative_func=["multiply","add","hypot","maximum"]

def mutation_permutations(tree1: Node) -> Node:  ##i have the problem that chainging a node changes all similar nodes
    offspring = deepcopy(tree1)  # single deepcopy

    
    #all_nodes = list(offspring.subtree)
    internal_nodes = [node for node in offspring.subtree if not node.is_leaf and node.arity>=2 and node.short_name not in commutative_func]  #maybe use non cummutative functions only 
    #print(internal_nodes)
    if (len(internal_nodes)>=1):
    

        target = gxgp_random.choice(internal_nodes)
        successors = target.successors
        succ1=deepcopy(successors[0])
        succ2=deepcopy(successors[1])

        target.successors=[succ2,succ1]
    

    return offspring


    """ if target_pointer == offspring:
        # Replacing root
        return new_node

    # Recursive replacement
    def replace(node):
        new_successors = []
        for child in node.successors:
            if child == target_pointer:
                new_successors.append(new_node)
            else:
                new_successors.append(replace(child))
        return Node(node._func, new_successors, name=node.short_name) """

    #return replace(offspring)


    
    """ else:
        # Replace function with another one of same arity
        original_arity = node.arity
        same_arity_ops = [op for op in operators if arity(op) == original_arity]
        if same_arity_ops:
            new_op = gxgp_random.choice(same_arity_ops)
            mutated_node = Node(new_op, node.successors)
            return offspring.replace_node(node, mutated_node) """

    return offspring


    """ while individual is None or len(individual) < n_nodes:
            op = gxgp_random.choice(self._operators)
            #params = gxgp_random.choices(pool, k=arity(op))
            individual = Node(op, params)
            pool.append(individual)
        return individual """
    
    return "ciao"
