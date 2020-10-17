import os
from dotenv import load_dotenv, find_dotenv

def main ():
    # find .env automagically by walking up directories until it's found
    dotenv_path = find_dotenv()

    # load up the entries as environment variables
    load_dotenv(dotenv_path)

    NEWSAPIKEY = os.environ.get("NEWSAPIKEY")
    MONGOUSERNAME = os.environ.get("MONGOUSERNAME")
    MONGOPASSWORD = os.environ.get("MONGOPASSWORD")
    MONGODB = os.environ.get("MONGODB")

    return "Variables={{NEWSAPIKEY={},MONGOUSERNAME={},MONGOPASSWORD={},MONGODB={}}}".format(NEWSAPIKEY, MONGOUSERNAME, MONGOPASSWORD, MONGODB)


if __name__ == "__main__":
    print(main())
    