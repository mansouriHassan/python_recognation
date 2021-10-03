import json
 
def parsing_file(fileName):
    # Opening JSON file
    f = open(fileName,)
    
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    print(data)
    # Iterating through the json
    # list
    for i in data['emp_details']:
        print(i)
    
    # Closing file
    f.close()