from contextlib import asynccontextmanager
from datetime import datetime
import os
import sys
from typing import Optional, List
import base64

from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
from fastapi import FastAPI, status, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


from dal import ToDoDAL, ListSummary, ToDoList
from dotenv import load_dotenv

load_dotenv(r"C:\Users\damil\AIgenda\.env")
COLLECTION_NAME = "new_todolists"
USERS_NAME = "walletaddresses"
# os.environ['MONGODB_URI'] = os.environ.get("MONGODB_URI")
os.environ['GOOGLE_API_KEY'] = os.environ.get("GOOGLE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
MONGO_DB_URI = os.environ.get("MONGODB_URI")
DEBUG = os.environ.get("DEBUG", "").strip().lower() in {"i", "true", "on", "yes"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup:
    client = AsyncIOMotorClient(
        MONGO_DB_URI
    )
    database = client.get_default_database()

    # ensure the database is available
    pong = await database.command("ping")
    if int(pong["ok"]) != 1:
        raise Exception("Cluster connection is not okay")
    todo_lists = database.get_collection(COLLECTION_NAME)
    users_lists = database.get_collection(USERS_NAME)
    app.todo_dal = ToDoDAL(todo_lists, users_lists)

    yield

    client.close()

model = llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
app = FastAPI(lifespan=lifespan, debug=DEBUG)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with a list of allowed origins if needed.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class User(BaseModel):
    wallet_address: str


class NewList(BaseModel):
    name: str
    from_img: bool = False
    img: Optional[str] = ""
    suggestion: Optional[str] = " "


class NewListResponse(BaseModel):
    id: str
    name: str


class item(BaseModel):
    items: list[str] = Field(description="the list of todo items")
    

@app.post("/api/users/", status_code=status.HTTP_201_CREATED)
async def create_user(user: User):
    result = await app.todo_dal.create_user(user.wallet_address)
    if not result:
        raise HTTPException(
            status_code=400,
            detail="User with this wallet address already exists")
    return {"wallet_address": result}


@app.get("/api/users/{wallet_address}/",)
async def get_user(wallet_address: str) -> User:
    user = await app.todo_dal.get_user(wallet_address)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/api/lists/")
async def get_all_lists() -> list[ListSummary]:
    return [i async for i in app.todo_dal.list_todo_lists()]


def get_todo_from_img(img):
    img_b64 = img
    structured_model = model.with_structured_output(item)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Reply with the a list of to-do items in the image provided."),
        ("user", [
            {"type": "text", "text": "{text}"},
            {"type": "image_url", "image_url": {"url": "{image}"}} if "{image}" else None
        ])
    ])

    # Create the chain
    chain = (
        RunnablePassthrough.assign(
            text=lambda x: x.get('text', ''),
            image=lambda x: x.get('image', None)
        )
        | prompt 
        | structured_model
    )
    result = chain.invoke({
        "image": f"data:image/jpeg;base64,{img_b64}",
        "text": "Describe this image"
    })
    return result.items


@app.post("/api/users/{wallet_address}/lists/",
          status_code=status.HTTP_201_CREATED)
async def create_todo_list(wallet_address,
                           new_list: NewList) -> NewListResponse:
    if new_list.img:
        items = get_todo_from_img(new_list.img
                                  )
        list_id = await app.todo_dal.create_todo_list(
                wallet_address=wallet_address,
                name=new_list.name,
                img=new_list.img,
                suggestion=new_list.suggestion,)
        for item in items:
            await app.todo_dal.create_item(
                wallet_address,
                list_id,
                item
            )
        return NewListResponse(
            id=list_id,
            name=new_list.name
        )
    return NewListResponse(
        id=await app.todo_dal.create_todo_list(
            wallet_address=wallet_address,
            name=new_list.name,
            img=new_list.img,
            suggestion=new_list.suggestion,
        ),
        name=new_list.name,
    )


@app.get("/api/users/{wallet_address}/lists/{list_id}/")
async def get_list(wallet_address: str, list_id: str) -> ToDoList:
    """Get a single to-do list for a user"""
    result = await app.todo_dal.get_todo_list(wallet_address=wallet_address,
                                           id=list_id)
    if result:
        return result
    else:
        raise HTTPException(
            status_code=404,
            detail="user has not created a list"
        )


@app.get("/api/users/{wallet_address}/lists/")
async def get_all_lists(wallet_address: str) -> List:
    """Get all to-do lists for a user"""
    return await app.todo_dal.get_users_list(wallet_address)


@app.delete("/api/users/{wallet_address}/lists/{list_id}/")
async def delete_list(wallet_address: str, list_id: str) -> bool:
    return await app.todo_dal.delete_todo_list(wallet_address, list_id)

@app.delete("/api/users/{wallet_address}/lists/")
async def delete_all_lists(wallet_address: str) -> bool:
    return await app.todo_dal.delete_all_lists(wallet_address)


class NewItem(BaseModel):
    label: str


class NewItemResponse(BaseModel):
    id: str
    label: str


@app.post(
    "/api/users/{wallet_address}/lists/{list_id}/items/",
    status_code=status.HTTP_201_CREATED,
)
async def create_item(wallet_address: str, list_id: str, new_item: NewItem) -> ToDoList:
    return await app.todo_dal.create_item(wallet_address, list_id,
                                          new_item.label)


@app.delete("/api/{wallet_address}/lists/{list_id}/items/{item_id}/")
async def delete_item(list_id: str, item_id: str) -> ToDoList:
    return await app.todo_dal.delete_item(list_id, item_id)


class ToDoItemUpdate(BaseModel):
    item_id: str
    checked_state: bool


@app.patch("/api/users/{wallet_address}/lists/{list_id}/checked_state/")
async def set_checked_state(list_id: str, update: ToDoItemUpdate) -> ToDoList:
    return await app.todo_dal.set_checked_state(
        list_id, update.item_id, update.checked_state
    )


async def create_smart_suggestion(todoitems):
    todo = [f"item: {item.label}, done: {item.checked}\n" for item in todoitems]
    todo = " ".join(todo)
    system_prompt = """You are a helpful, kind and homely assistant that helps users with their todo lists. 
    You are to provide useful suggestions and a guide to help a user complete their tasks for the day. You are to provide the user with useful responses and strategies to complete the todolists provided to you.
    """
    model = ChatOpenAI(
        model="gpt-4o"
    )

    # Text-only input
    input_ = {"messages": [HumanMessage(content=todo)]}

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{messages}")]
    )

    # Create the runnable chain
    assistant_runnable = RunnablePassthrough.assign(response=prompt | model)

    # Invoke the runnable
    try:
        result = assistant_runnable.invoke(input_)
    except Exception as e:
        print(f"An error occurred: {e}")

    return result['response'].content


@app.post("/api/users/{wallet_address}/lists/{list_id}/smart_suggestions/")
async def get_smart_suggestions(
    wallet_address: str,
    list_id: str,
):
    todo = await app.todo_dal.get_todo_list(wallet_address, list_id)
    if not todo:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized request"
        )
    todo_items = todo.items
    suggestion = await create_smart_suggestion(todo_items)
    result = await app.todo_dal.add_suggestions(list_id, suggestion)
    return result


class DummyResponse(BaseModel):
    id: str
    when: datetime


@app.get("/api/dummy")
async def get_dummy() -> DummyResponse:
    return DummyResponse(
        id=str(ObjectId()),
        when=datetime.now(),
    )


def main(argv=sys.argv[1:]):
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=3001, reload=False)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()