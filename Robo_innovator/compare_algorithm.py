def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union

def find_most_similar(reference_list, data):
    max_similarity = -1
    most_similar_list = None

    for lst in data:
        similarity = jaccard_similarity(reference_list, lst)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_list = lst

    return most_similar_list

data = [[1, 1, 2], [1, 3, 5], [1, 3, 9], [2, 4, 9], [2, 4, 9], [2, 4, 9], [2, 4, 9], [2, 4, 9], [2, 4, 9]]
reference_list = [1, 3, 9]

most_similar_list = find_most_similar(reference_list, data)
print("Most similar list:", most_similar_list)
