# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq
from operator import ne
import os
import pickle
import math
from networkx.algorithms.clique import enumerate_all_cliques

from networkx.algorithms.similarity import optimal_edit_paths







class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        popped_node = heapq.heappop(self.queue)
        return (popped_node[0], popped_node[2])           # changes made


    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """
        for a_node in self.queue:
            if(node == a_node):
                self.queue.remove(node)


    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        heap_counter = len(self.queue)         
        insert_node = (node[0], heap_counter, node[1])
        heapq.heappush(self.queue, insert_node)
       
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]

####################### Helper functions for BFS ###############################
def node_in_lists_bfs(node, frontier, explored_nodes, dummy_frontier):
    for i in dummy_frontier:
        if(node == i[1]):           
            return True
    for i in frontier:
        if(node == i[1]):           
            return True
    for i in explored_nodes:
        if(node == i[1]):           
            return True
    return False


def find_path_bfs(path, explored_nodes, test_node, start):
    for i in range(0, len(explored_nodes)):
        if(explored_nodes[i][1] == test_node):                     
            path.append(explored_nodes[i][1])                       
            if(explored_nodes[i][1] == start):
                return
            find_path_bfs(path, explored_nodes, explored_nodes[i][2], start)   # changes (done)
################################################################################
def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if(start == goal):
        return []
    else:
        frontier = []
        frontier.append((0 ,start, ''))        
        dummy_frontier = []
        explored_nodes = []
        neighbors_list = []
        goal_test = False

        while(True):
            # goal test
            curr_node = frontier[0]         
            neighbors = graph[curr_node[1]]     

            for node in neighbors:
                if(node == goal):
                    goal_test = True
                    explored_nodes.append((curr_node[0] + 1, node, curr_node[1]))
                # check if node is in frontier or explored
                if(node_in_lists_bfs(node, frontier, explored_nodes, dummy_frontier)): 
                    continue
                neighbors_list.append(node)

            for node in neighbors_list:                            
                dummy_frontier.append((curr_node[0] + 1, node, curr_node[1]))     
            
            if(goal_test):
                explored_nodes.append(curr_node)
                break

            neighbors_list.clear()
            frontier.remove(curr_node)
            explored_nodes.append(curr_node)

            if(frontier == []):
                for node in dummy_frontier:
                    frontier.append(node)                            
                dummy_frontier.clear()
                frontier.sort()


    path = []
    find_path_bfs(path, explored_nodes, goal, start)
    path.reverse()
    print(path)
    return path

def find_path_ucs(path, explored_nodes, node, start):
    for i in explored_nodes:
        if(node == i[0][0]):
            path.append(node)
            if(node == start):
                return
            find_path_ucs(path, explored_nodes, i[0][1], start)

####################### Helper functions for UCS ###############################
# def check_parent(node, curr_node):
#         if(node[1][0] == curr_node[1][1]):
#             return True
#         return False
    
def check_better_path_ucs(node, frontier_list):
    for frontier_node in frontier_list:
        if(node[1][0] == frontier_node[2][0]):
            if(node[0] < frontier_node[0]):
                ind = frontier_list.index(frontier_node)
                frontier_list.insert(ind, (node[0], frontier_node[1], node[1]))
                frontier_list.remove(frontier_node)

                return True
            else:
                return True
        else:
            continue
    return False

def check_in_explored_ucs(node, explored_list):
    for explored_node in explored_list:
        if(explored_node[1][0] == node[1][0]):
            return True
    return False
#################################################################################

def uniform_cost_search(graph, start, goal):

    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if(start == goal):
        return []
    else:
        frontier = PriorityQueue()
        frontier.append((0 ,(start, 'null')))       
        explored_nodes = []
        neighbors_list = []

        while(True):
            curr_node = frontier.pop()
            # put goal test
            if(curr_node[1][0] == goal):
                explored_nodes.append(curr_node)
                break

            neighbors = graph[curr_node[1][0]]

            for node in neighbors:
                if(node == curr_node[1][1]):
                    continue
                neighbors_list.append(node)

            for node in neighbors_list:
                edge_weight = graph.get_edge_weight(curr_node[1][0], node)
                new_edge_weight = curr_node[0] + edge_weight
                parent = curr_node[1][0]
                new_node = (new_edge_weight, (node, parent))

                # 1. check if in explored
                if(check_in_explored_ucs(new_node, explored_nodes)):
                    continue               

                # 2. check so that if better path available to be put on frontier
                if(check_better_path_ucs(new_node, frontier.queue)):
                    continue
                
                frontier.append(new_node)
            
            neighbors_list.clear()
            explored_nodes.append(curr_node)

    test_node = goal
    path = [goal]

    for i in path:
        if(i == start):
            break
        for node in explored_nodes:
            if(i == node[1][0]):
                path.append(node[1][1])
                break

    path.reverse()
    return path


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    v_pos = graph.nodes[v]['pos']
    v_pos_x = v_pos[0]
    v_pos_y = v_pos[1]

    goal_pos = graph.nodes[goal]['pos']
    goal_pos_x = goal_pos[0]
    goal_pos_y = goal_pos[1]

    dist = math.sqrt((v_pos_x - goal_pos_x)**2 + (v_pos_y - goal_pos_y)**2)
    return dist

def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if(start == goal):
        return []
    else:
        frontier = PriorityQueue()
        frontier.append((0 ,(start, 'null')))       
        explored_nodes = []
        neighbors_list = []

        while(True):
            curr_node = frontier.pop()
            # put goal test
            if(curr_node[1][0] == goal):
                explored_nodes.append(curr_node)
                break

            neighbors = graph[curr_node[1][0]]                  # expand the current node

            for node in neighbors:
                if(node == curr_node[1][1]):
                    continue
                neighbors_list.append(node)                     # get neighbors in a list

            for node in neighbors_list:
                parent = curr_node[1][0]
                edge_weight = graph.get_edge_weight(curr_node[1][0], node)
                node_dist = euclidean_dist_heuristic(graph, node, goal)
                parent_dist = euclidean_dist_heuristic(graph, curr_node[1][0], goal)
                new_edge_weight = curr_node[0] + edge_weight + node_dist - parent_dist

                new_node = (new_edge_weight, (node, parent))

                # 1. check if in explored
                if(check_in_explored_ucs(new_node, explored_nodes)):
                    continue
              
                # 2. check so that if better path available to be put on frontier
                if(check_better_path_ucs(new_node, frontier.queue)):
                    continue
                
                frontier.append(new_node)
            
            neighbors_list.clear()
            explored_nodes.append(curr_node)

    test_node = goal
    path = [goal]

    for i in path:
        if(i == start):
            break
        for node in explored_nodes:
            if(i == node[1][0]):
                path.append(node[1][1])
                break

    path.reverse()
    return path

####################### Helper functions for Bi_UCS ###############################
def min_cost_frontier(frontier_queue, origin):
    min_cost = -1
    for frontier_node in frontier_queue:
        if(origin == frontier_node[2][2]):
            if(min_cost == -1):
                min_cost = frontier_node[0]
            else:
                if(frontier_node[0] < min_cost):
                    min_cost = frontier_node[0]
    return min_cost


def find_path(path, node, goal, explored_nodes):
    if(goal == node[1][0]):
        return
    else:
        path.append(node[1][1])
        for i in path:
            if(i == goal):
                break
            for a_node in explored_nodes:
                if(i == a_node[1][0]):
                    path.append(a_node[1][1])
                    break
    return
  
def make_path_bi(path, node1, node2, start, goal, explored):
    curr_path_cost = path[0]
    curr_path = path[1]
    new_path_cost = node1[0] + node2[0]
   
    if(new_path_cost > curr_path_cost and curr_path_cost != -1):
        return
    else:
        node1_path = []
        node2_path = []
        if(node1[1][2] == 'S'):
            find_path(node1_path, node1, start, explored)
            node1_path.reverse()
            node1_path.append(node1[1][0])
            find_path(node2_path, node2, goal, explored)
            new_path = node1_path + node2_path
        else:
            find_path(node2_path, node2, start, explored)
            node2_path.reverse()
            node2_path.append(node2[1][0])
            find_path(node1_path, node1, goal, explored)
            new_path = node2_path + node1_path

        path[0] = new_path_cost
        path[1] = new_path






#########################################################################################
class Node_obj(object):
    def __init__(self):
        value = (0, ('', '', ''))

    def Priority(self):
        return self.value[0]
    def Node(self):
        return self.value[1][0]
    def Parent(self):
        return self.value[1][1]
    def Origin(self):
        return self.value[1][2]

def check_in_explored_bi_ucs(new_node, explored):
    for a_node in explored:
        if(a_node[1][0] == new_node.Node()):
            return True
    return False

def check_better_path_bi_ucs(new_node, frontier):
    for a_node in frontier.queue:
        count = 0
        if(a_node[2][0] == new_node.Node() and a_node[0] > new_node.Priority()):
            frontier.queue.remove(a_node)
            frontier.queue.insert(count, (new_node.Priority(), count, (new_node.Node(), new_node.Parent(), new_node.Origin()) ) )
            return True
        count += 1
    return False
   

def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    
    # TODO: finish this function!
    # test trivial case1
    if(start == goal):
        return []                           # when start and goal nodes are same
    
    # test tivial case2
    path = []
    path_cost = -1
    explored_s = []
    explored_g = []
    neighbors_list = []
    frontier_s = PriorityQueue()
    frontier_g = PriorityQueue()

    start_node = Node_obj()
    start_node.value = (0, (start, 'null', 'S'))
    frontier_s.append(start_node.value)

    goal_node = Node_obj()
    goal_node.value = (0, (goal, 'null', 'G'))
    frontier_g.append(goal_node.value)
 
    # first pop node from the S frontier
    curr_node = Node_obj()
    new_node = Node_obj()
    curr_node.value = frontier_s.pop()
    neighbors = graph[curr_node.Node()]
    explored_s.append(curr_node.value)

    for node in neighbors:
        neighbors_list.append(node)
        if(node == goal):
            return [start, goal]                # when start and goal are adjacent
    
    for node in neighbors_list:
        edge_weight = graph.get_edge_weight(curr_node.Node(), node)
        new_edge_weight = curr_node.Priority() + edge_weight
        parent = curr_node.Node()
        new_node.value = (new_edge_weight, (node, parent, curr_node.Origin()))
        frontier_s.append(new_node.value)
    neighbors_list.clear()

    curr_node.value = frontier_g.pop()
    neighbors = graph[curr_node.Node()]
    explored_g.append(curr_node.value)

    for node in neighbors:
        neighbors_list.append(node)
    
    for node in neighbors_list:
        edge_weight = graph.get_edge_weight(curr_node.Node(), node)
        new_edge_weight = curr_node.Priority() + edge_weight
        parent = curr_node.Node()
        new_node.value = (new_edge_weight, (node, parent, curr_node.Origin()))
        frontier_g.append(new_node.value)
    neighbors_list.clear()

    prev_front = 'G'
    while(True):
        if(prev_front == 'G'):
            if(frontier_s.queue != []):
                # pop node from frontier s
                prev_front = 'S'
                curr_node.value = frontier_s.pop()
                explored_s.append(curr_node.value)
                # check and remove the repeating node in the other frontier
                for a_node in frontier_g.queue:
                    if(a_node[2][0] == curr_node.Node()):
                        frontier_g.remove(a_node)
                
                neighbors = graph[curr_node.Node()]
                for node in neighbors:
                    if(node == curr_node.Parent()):
                        continue
                    neighbors_list.append(node)

                for node in neighbors_list:
                    edge_weight = graph.get_edge_weight(curr_node.Node(), node)
                    new_edge_weight = curr_node.Priority() + edge_weight
                    parent = curr_node.Node()
                    new_node.value = (new_edge_weight, (node, parent, curr_node.Origin()))
                    
                    # - if neighbor in the explored set of current - continue
                    if(check_in_explored_bi_ucs(new_node, explored_s)):
                        continue
                    # - if neibhor is in the explored set of the other frontier make path
                    for a_node in explored_g:
                        if(a_node[1][0] == new_node.Node()):
                            s_path = []
                            g_path = []
                            new_path_cost = new_node.Priority() + a_node[0]
                            find_path(s_path, new_node.value, start, explored_s)
                            find_path(g_path, a_node, goal, explored_g)
                            s_path.reverse()
                            new_path = s_path + [new_node.Node()] + g_path
                            if(path_cost == -1):
                                path = new_path
                                path_cost = new_path_cost
                            else:
                                if(new_path_cost < path_cost):
                                    path = new_path
                                    path_cost = new_path_cost                        
                        continue
                    # - if new_node in the frontier of the current - check if you can replace, otherwise continue
                    if(check_better_path_bi_ucs(new_node, frontier_s)):
                        continue
                    frontier_s.append(new_node.value)
                neighbors_list.clear()
            else:
                prev_front = 'S'

        else:
            if(prev_front == 'S'):
                if(frontier_g.queue != []):
                    # pop node from frontier s
                    prev_front = 'G'
                    curr_node.value = frontier_g.pop()
                    explored_g.append(curr_node.value)
                    # check and remove the repeating node in the other frontier
                    for a_node in frontier_s.queue:
                        if(a_node[2][0] == curr_node.Node()):
                            frontier_g.remove(a_node)
                    
                    neighbors = graph[curr_node.Node()]
                    for node in neighbors:
                        if(node == curr_node.Parent()):
                            continue
                        neighbors_list.append(node)

                    for node in neighbors_list:
                        edge_weight = graph.get_edge_weight(curr_node.Node(), node)
                        new_edge_weight = curr_node.Priority() + edge_weight
                        parent = curr_node.Node()
                        new_node.value = (new_edge_weight, (node, parent, curr_node.Origin()))
                        
                        # - if new_node in the explored set of current - continue
                        if(check_in_explored_bi_ucs(new_node, explored_g)):
                            continue
                        # - if new_node is in the explored set of the other frontier make path
                        for a_node in explored_s:
                            if(a_node[1][0] == new_node.Node()):
                                s_path = []
                                g_path = []
                                new_path_cost = new_node.Priority() + a_node[0]
                                find_path(g_path, new_node.value, goal, explored_g)
                                find_path(s_path, a_node, start, explored_s)
                                s_path.reverse()
                                new_path = s_path + [new_node.Node()] + g_path
                                if(path_cost == -1):
                                    path = new_path
                                    path_cost = new_path_cost
                                else:
                                    if(new_path_cost < path_cost):
                                        path = new_path
                                        path_cost = new_path_cost                        
                            continue
                        # - if new_node in the frontier of the current - check if you can replace, otherwise continue
                        if(check_better_path_bi_ucs(new_node, frontier_g)):
                            continue
                        frontier_g.append(new_node.value)
                    neighbors_list.clear()
                else:
                    prev_front = 'G'

        if(frontier_s.queue == [] and frontier_g.queue == []):
            break
        if(path_cost != -1):
            if(frontier_s.queue == []):
                cost = min_cost_frontier(frontier_g.queue, 'G')
                if(cost > path_cost):
                    break
            elif(frontier_g.queue == []):
                cost = min_cost_frontier(frontier_s.queue, 'S')
                if(cost > path_cost):
                    break
            else:
                cost = min_cost_frontier(frontier_s.queue, 'S') + min_cost_frontier(frontier_g.queue, 'G')
                if(cost > path_cost):
                    break
    print(path)
    return path
       


###################################### HELPER FUNCTIONS FOR BI-DIREC_A* #########################
      
class Node_obj_star(object):
    def __init__(self):
        value = (0, ('', '', '', 0))

    def Priority(self):
        return self.value[0]
    def Node(self):
        return self.value[1][0]
    def Parent(self):
        return self.value[1][1]
    def Origin(self):
        return self.value[1][2]
    def Heuristic(self):
        return self.value[1][3]

def check_in_explored_bi_astar(new_node, explored):
    for a_node in explored:
        if(a_node[1][0] == new_node.Node()):
            return True
    return False

def check_better_path_bi_astar(new_node, frontier):
    for a_node in frontier.queue:
        count = 0
        if(a_node[2][0] == new_node.Node() and a_node[0] > new_node.Priority()):
            frontier.queue.remove(a_node)
            frontier.queue.insert(count, (new_node.Priority(), count, (new_node.Node(), new_node.Parent(), new_node.Origin(), new_node.Heuristic()) ) )
            return True
        count += 1
    return False

def dist_between_points(a_pos, b_pos):              # a_pos and b_pos must be lists containing [x,y]
    return math.sqrt( ( a_pos[0] - b_pos[0] ) ** 2 + ( a_pos[1] - b_pos[1] ) ** 2  )

def avg_pos(frontier, prev_avg_pos):                 # returns the average position of the frontier
    if(frontier.queue == []):
        return prev_avg_pos
    else:
        sum = 0
        count = 0
        x_sum = 0
        y_sum = 0
        for node in frontier.queue:
            x_sum += node[2][3]
            y_sum += node[2][3]
            count += 1
        return [x_sum/count, y_sum/count]

def min_cost_frontier_a_star(frontier):
    min_cost = -1
    for a_node in frontier:
        if(min_cost == -1):
            min_cost = a_node[0] - a_node[2][3]
        else:
            if(a_node[2][3] < min_cost):
                min_cost = a_node[2][3]
    return min_cost


#######################################################################################################
def bidirectional_a_star(graph, start, goal,
                         heuristic=dist_between_points):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # test trivial case1
    if(start == goal):
        return []                           # when start and goal nodes are same
    
    # test tivial case2
       
    path = []
    path_cost = -1
    explored_s = []
    explored_g = []
    neighbors_list = []

    frontier_s = PriorityQueue()
    frontier_g = PriorityQueue()

    start_node = Node_obj_star()
    start_node.value = (0, (start, 'null', 'S', 0))
    frontier_s.append(start_node.value)

    goal_node = Node_obj_star()
    goal_node.value = (0, (goal, 'null', 'G', 0))
    frontier_g.append(goal_node.value)
 
    # first pop node from the S frontier
    curr_node = Node_obj_star()
    new_node = Node_obj_star()
    curr_node.value = frontier_s.pop()
    neighbors = graph[curr_node.Node()]
    explored_s.append(curr_node.value)

    for node in neighbors:
        neighbors_list.append(node)
        if(node == goal):
            return [start, goal]                # when start and goal are adjacent
    
    for node in neighbors_list:
        edge_weight = graph.get_edge_weight(curr_node.Node(), node)
        heuristic = int(euclidean_dist_heuristic(graph, node, goal))
        new_edge_weight = edge_weight + curr_node.Priority() - curr_node.Heuristic() + heuristic
        parent = curr_node.Node()
        new_node.value = (new_edge_weight, (node, parent, curr_node.Origin(), heuristic))
        frontier_s.append(new_node.value)
    neighbors_list.clear()

    curr_node.value = frontier_g.pop()
    neighbors = graph[curr_node.Node()]
    explored_g.append(curr_node.value)

    for node in neighbors:
        neighbors_list.append(node)
    
    for node in neighbors_list:
        edge_weight = graph.get_edge_weight(curr_node.Node(), node)
        heuristic = int(euclidean_dist_heuristic(graph, start, node))
        new_edge_weight = edge_weight + curr_node.Priority() - curr_node.Heuristic() + heuristic
        parent = curr_node.Node()
        new_node.value = (new_edge_weight, (node, parent, curr_node.Origin(), heuristic))
        frontier_g.append(new_node.value)
    neighbors_list.clear()

    prev_front = 'G'
    while(True):
        if(prev_front == 'G'):
            if(frontier_s.queue != []):
                # pop node from frontier s
                prev_front = 'S'
                curr_node.value = frontier_s.pop()
                explored_s.append(curr_node.value)
                # check and remove the repeating node in the other frontier
                for a_node in frontier_g.queue:
                    if(a_node[2][0] == curr_node.Node()):
                        frontier_g.remove(a_node)
                
                neighbors = graph[curr_node.Node()]
                for node in neighbors:
                    if(node == curr_node.Parent()):
                        continue
                    neighbors_list.append(node)

                for node in neighbors_list:
                    edge_weight = graph.get_edge_weight(curr_node.Node(), node)
                    heuristic = int(euclidean_dist_heuristic(graph, node, goal))
                    new_edge_weight = edge_weight + curr_node.Priority() - curr_node.Heuristic() + heuristic
                    parent = curr_node.Node()
                    new_node.value = (new_edge_weight, (node, parent, curr_node.Origin(), heuristic))
                    
                    # - if neighbor in the explored set of current - continue
                    if(check_in_explored_bi_astar(new_node, explored_s)):
                        continue
                    # - if neibhor is in the explored set of the other frontier make path
                    check = False
                    for a_node in explored_g:
                        if(a_node[1][0] == new_node.Node()):
                            check = True
                            s_path = []
                            g_path = []
                            new_path_cost = new_node.Priority() - new_node.Heuristic() + a_node[0] - a_node[1][3]
                            find_path(s_path, new_node.value, start, explored_s)
                            find_path(g_path, a_node, goal, explored_g)
                            s_path.reverse()
                            new_path = s_path + [new_node.Node()] + g_path
                            if(path_cost == -1):
                                path = new_path
                                path_cost = new_path_cost
                                break
                            else:
                                if(new_path_cost < path_cost):
                                    path = new_path
                                    path_cost = new_path_cost 
                                    break
                                else:
                                    break                       
                    if(check):
                        continue

                    # - if new_node in the frontier of the current - check if you can replace, otherwise continue
                    if(check_better_path_bi_astar(new_node, frontier_s)):
                        continue
                    frontier_s.append(new_node.value)
                neighbors_list.clear()
            else:
                prev_front = 'S'

        else:
            if(prev_front == 'S'):
                if(frontier_g.queue != []):
                    # pop node from frontier s
                    prev_front = 'G'
                    curr_node.value = frontier_g.pop()
                    explored_g.append(curr_node.value)
                    # check and remove the repeating node in the other frontier
                    for a_node in frontier_s.queue:
                        if(a_node[2][0] == curr_node.Node()):
                            frontier_g.remove(a_node)
                    
                    neighbors = graph[curr_node.Node()]
                    for node in neighbors:
                        if(node == curr_node.Parent()):
                            continue
                        neighbors_list.append(node)

                    for node in neighbors_list:
                        edge_weight = graph.get_edge_weight(curr_node.Node(), node)
                        heuristic = int(euclidean_dist_heuristic(graph, node, start))
                        new_edge_weight = edge_weight + curr_node.Priority() - curr_node.Heuristic() + heuristic
                        parent = curr_node.Node()
                        new_node.value = (new_edge_weight, (node, parent, curr_node.Origin(), heuristic))
                        
                        # - if new_node in the explored set of current - continue
                        if(check_in_explored_bi_astar(new_node, explored_g)):
                            continue
                        # - if new_node is in the explored set of the other frontier make path
                        check = False
                        for a_node in explored_s:
                            if(a_node[1][0] == new_node.Node()):
                                check = True
                                s_path = []
                                g_path = []
                                new_path_cost = new_node.Priority() - new_node.Heuristic() + a_node[0] - a_node[1][3]
                                find_path(g_path, new_node.value, goal, explored_g)
                                find_path(s_path, a_node, start, explored_s)
                                s_path.reverse()
                                new_path = s_path + [new_node.Node()] + g_path
                                if(path_cost == -1):
                                    path = new_path
                                    path_cost = new_path_cost
                                    break
                                else:
                                    if(new_path_cost < path_cost):
                                        path = new_path
                                        path_cost = new_path_cost   
                                        break
                                    else:
                                        break
                        if(check):
                            continue
                                                                 
                        # - if new_node in the frontier of the current - check if you can replace, otherwise continue
                        if(check_better_path_bi_astar(new_node, frontier_g)):
                            continue
                        frontier_g.append(new_node.value)
                    neighbors_list.clear()
                else:
                    prev_front = 'G'

        if(frontier_s.queue == [] and frontier_g.queue == []):
            break
        if(path_cost != -1):
            if(frontier_s.queue == []):
                cost = min_cost_frontier_a_star(frontier_g)
                if(cost > path_cost):
                    break
            elif(frontier_g.queue == []):
                cost = min_cost_frontier_a_star(frontier_s)
                if(cost > path_cost):
                    break
            else:
                cost = min_cost_frontier_a_star(frontier_s) + min_cost_frontier_a_star(frontier_g)
                if(cost > path_cost):
                    break
    print(path)
    return path    


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Manan Patel"



def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula