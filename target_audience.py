import json

def who_is_my_client():
    client = input("Who is your client? ")
    return client

def what_does_he_live_by():
    client = input("What does he live by? ")
    return client

def what_pains_does_he_have():
    client = input("Who does pains he have? ")
    return client

def target_audience():
    with open('target_audience.json', 'r') as file:
        data = json.load(file)
    data["client"].append(who_is_my_client())
    data["interests"].append(what_does_he_live_by())
    data["pains"].append(what_pains_does_he_have())
    # data = {"client": [], "interests": [], "pains": []}
    # data = {"client": who_is_my_client(), "interests": what_does_he_live_by(), "pains": what_pains_does_he_have()}
    with open('target_audience.json', 'w') as outfile:
        json.dump(data, outfile)

if __name__ == '__main__':
    target_audience()
