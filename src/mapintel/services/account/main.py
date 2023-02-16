from fastapi import FastAPI
from tortoise import fields, models

app = FastAPI()


class User(models.Model):
    
    username = fields.TextField()
    password = fields.TextField()
    created_at = fields.DatetimeField(auto_now_add=True)

    def __str__(self):
        return self.username

@app.get('/index')
async def index():
    return {'message': 'Hello'}