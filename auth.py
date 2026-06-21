users={

"admin":"1234"

}

def login(

username,
password

):

    if username in users:

        return users[
            username
        ]==password

    return False