import math

side = ['top-l', 'top-c', 'top-r', 'bottom-l', 'bottom-c', 'bottom-r', 'left', 'right']

def distance_veh(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def process_distances(data):
    for i, obj1 in enumerate(data):
        min_distances = {}
        min_distance_details = {}

        for j, obj2 in enumerate(data):
            if i != j:
                distances = {}
                coord_pairs = {}
                for s1 in side:
                    for s2 in side:
                        new_key = s1 + "-" + s2
                        distances[new_key] = round(distance_veh(obj1[s1], obj2[s2]), 3)
                        coord_pairs[new_key] = (obj1[s1], obj2[s2])
                
                min_key = min(distances, key=distances.get)
                min_distances[obj2['id']] = distances[min_key]
                min_distance_details[obj2['id']] = {
                    "key": min_key,
                    "start_point": coord_pairs[min_key][0],
                    "end_point": coord_pairs[min_key][1]
                }
                # min_distances[obj2['id']] = min(distances.values())

        sorted_data = sorted(min_distances.items(), key=lambda x: x[1])
        # top_5 = sorted_data[:5]
        # obj1['distances_to_other_ids'] = dict(sorted(dict(top_5).items()))
        obj1['distances_to_other_ids'] = dict(sorted_data)
        obj1['distance_details'] = {k: min_distance_details[k] for k, _ in sorted_data}

    return data
