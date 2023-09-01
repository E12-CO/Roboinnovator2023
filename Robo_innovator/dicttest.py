my_dict = {"Java": 100, "Python": 112, "C": 11}

# One-liner
print("One line Code Key value:", [key for key, value in my_dict.items() if value == max(my_dict.values())][0])
