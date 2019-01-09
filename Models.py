#importing peewee
from peewee import *
#connecting to SqliteDatabase
db=SqliteDatabase('Dataset/user_credentials.db')

#Table user
class User(Model):
    user_id=CharField()
    username=CharField(unique=True)
    email=CharField()
    password=CharField()

    class Meta:
        database=db
