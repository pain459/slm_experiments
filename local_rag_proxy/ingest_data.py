# ingest_data.py
import requests

url = "http://localhost:8000/ingest"

# Sample curriculum data (Replace this with your actual Python notes later!)
documents = [
    {
        "doc_id": "lesson_1_decorators",
        "text": "In Python, a decorator is a design pattern that allows you to modify the behavior of a function or class. Decorators are usually called before the definition of a function you want to decorate. To create a decorator, you define a wrapper function that takes another function as an argument."
    },
    {
        "doc_id": "lesson_2_asyncio",
        "text": "Asyncio is a library in Python to write concurrent code using the async/await syntax. It is used as a foundation for multiple Python asynchronous frameworks that provide high-performance network and web-servers, database connection libraries, and more. Use 'async def' to define a coroutine."
    },
    {
        "doc_id": "lesson_3_context_managers",
        "text": "Context managers in Python allow for setup and cleanup actions for blocks of code. The most common way to write a context manager is using the 'with' statement. You can implement a context manager by creating a class with '__enter__' and '__exit__' methods, or by using the '@contextmanager' decorator from the 'contextlib' module."
    }
]

for doc in documents:
    response = requests.post(url, json=doc)
    print(response.json())