import math

side = ['top-l', 'top-c', 'top-r', 'bottom-l', 'bottom-c', 'bottom-r', 'left', 'right']

def distance_veh(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def process_distances(data):
    for i, obj1 in enumerate(data):
        min_distances = {}
        for j, obj2 in enumerate(data):
            if i != j:
                distances = {}
                for s1 in side:
                    for s2 in side:
                        new_key = s1 + "-" + s2
                        distances[new_key] = round(distance_veh(obj1[s1], obj2[s2]), 3)
                min_distances[obj2['id']] = min(distances.values())
        obj1['distances_to_other_ids'] = min_distances
    return data