from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError
import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr

from uuid import uuid4

# class ListSummary(BaseModel):
#   id: str
#   name: str
#   item_count: int


#   @staticmethod
#   def from_doc(doc) -> "ListSummary":
#       return ListSummary(
#           id=str(doc["_id"]),
#           name=doc["name"],
#           item_count=doc["item_count"],
#       )
class ListSummary(BaseModel):
    id: str
    name: str
    item_count: int

    @staticmethod
    def from_doc(doc) -> "ListSummary":
        return ListSummary(
            id=str(doc["_id"]),
            name=doc["name"],
            item_count=len(doc.get("items", [])),  # Fix item count calculation
        )


class ToDoListItem(BaseModel):
    id: str
    label: str
    checked: bool

    @staticmethod
    def from_doc(item) -> "ToDoListItem":
        return ToDoListItem(
            id=item["id"],
            label=item["label"],
            checked=item["checked"],
        )


class User(BaseModel):
    wallet_address: str
    email: EmailStr

    @staticmethod
    def from_doc(doc):
        return User(
            wallet_address=doc['_id'],
            email=doc['email']
        )


class ToDoList(BaseModel):
    id: str
    name: str
    items: list[ToDoListItem]
    img: Optional[str]
    suggestion: Optional[str]

    @staticmethod
    def from_doc(doc) -> "ToDoList":
        return ToDoList(
            id=str(doc["_id"]),
            name=doc["name"],
            img=doc.get("img"),
            suggestion=doc.get("suggestion", " "),
            items=[
                ToDoListItem.from_doc(item)
                for item in doc["items"]
                if item in doc["items"]
            ],
        )


class ToDoDAL:
    def __init__(
        self,
        todo_collection: AsyncIOMotorCollection,
        users_collection: AsyncIOMotorCollection,
    ):
        """Database Interface Class"""
        self._todo_collection = todo_collection
        self._users_collection = users_collection

    async def create_user(self, user: User, session=None) -> str:
        try:
            user = await self._users_collection.insert_one(
                {"_id": user.wallet_address, "lists": [], "email":user.email}, session=session
            )
            if user:
                user['wallet_address'] = str(user["_id"])
                user['email'] = str(user['email'])
        except DuplicateKeyError:
            return None
        return 

    async def get_user(self, wallet_address: str, session=None):
        user = await self._users_collection.find_one(
            {"_id": wallet_address}, session=session
        )
        if user:
            user["wallet_address"] = str(user["_id"])
        return user

    async def list_todo_lists(self, session=None):
        async for doc in self._todo_collection.find(
            {},
            projection={
                "name": 1,
                "item_count": {"$size": "$items"},
            },
            sort={"name": 1},
            session=session,
        ):
            yield ListSummary.from_doc(doc)

    async def create_todo_list(
        self,
        wallet_address: str,
        name: str,
        img: str = None,
        suggestion: str = None,
        session=None,
    ) -> str:
        response = await self._todo_collection.insert_one(
            {"name": name, "items": [], "img": img, "suggestion": suggestion},
            session=session,
        )
        list_id = response.inserted_id
        await self._users_collection.update_one(
            {"_id": wallet_address}, {"$addToSet": {"lists": list_id}}, upsert=True
        )
        return str(response.inserted_id)

    async def get_users_list(self, wallet_address: str) -> List[ToDoList]:
        user = await self._users_collection.find_one({"_id": wallet_address})
        if not user:
            return []
        # Fetch the lists using list_ids
        list_ids = user.get("lists", [])
        docs = await self._todo_collection.find(
            {"_id": {"$in": list_ids}}
        ).to_list(None)
        return [ToDoList.from_doc(doc) for doc in docs]

    async def get_todo_list(
        self, wallet_address: str, id: str | ObjectId, session=None
    ) -> ToDoList:
        user = await self._users_collection.find_one(
            {"_id": wallet_address}, session=session
        )
        if not user:
            return None
        
        list_id = ObjectId(id)
        if list_id not in user.get("lists", []):
            return None

        doc = await self._todo_collection.find_one(
            {"_id": ObjectId(id)},
            session=session,
        )
        return ToDoList.from_doc(doc)

    #   async def delete_todo_list(self, id: str | ObjectId, session=None) -> bool:
    #       response = await self._todo_collection.delete_one(
    #           {"_id": ObjectId(id)},
    #           session=session,
    #       )
    #       return response.deleted_count == 1

    async def delete_all_lists(self, wallet_address: str, session=None):
        user = await self._users_collection.find_one(
            {"_id": wallet_address}, session=session            
        )
        if not user:
            return False
        list_ids = user.get("lists", [])
        for id in list_ids:
            response = await self._todo_collection.delete_one(
                {"_id" : id}, session=session
            )
            if response.deleted_count == 1:
                await self._users_collection.update_one(
                    {"_id": wallet_address}, {"$pull": {"lists": id}}, session=session
                )
        return True

    async def delete_todo_list(
        self, wallet_address: str, id: str | ObjectId, session=None
    ) -> bool:
        # Retrieve the user document using the wallet address
        user = await self._users_collection.find_one(
            {"_id": wallet_address}, session=session
        )
        if not user:
            # User not found
            return False

        # Convert id to ObjectId if necessary
        list_id = ObjectId(id) if not isinstance(id, ObjectId) else id

        # Ensure the list_id is associated with this user
        if list_id not in user.get("lists", []):
            # The to-do list doesn't belong to this user
            return False

        # Delete the to-do list
        response = await self._todo_collection.delete_one(
            {"_id": list_id}, session=session
        )
        if response.deleted_count == 1:
            # Remove the list reference from the user's document
            await self._users_collection.update_one(
                {"_id": wallet_address}, {"$pull": {"lists": list_id}},
                session=session
            )
            return True

    async def create_item(
        self,
        wallet_address: str,
        list_id: str | ObjectId,
        label: str,
        session=None,
    ) -> ToDoList | None:
        # Convert list_id to ObjectId if needed
        list_obj_id = ObjectId(list_id) if not isinstance(list_id, ObjectId) else list_id

        # Verify that the user exists and that the list is associated with the user
        user = await self._users_collection.find_one({"_id": wallet_address},
                                                     session=session)
        if not user:
            # User not found
            return None

        # Check if the list_id is in the user's list of to-do lists.
        if list_obj_id not in user.get("lists", []):
            # The specified list does not belong to the given wallet address.
            return None

        # Add the new item to the to-do list
        result = await self._todo_collection.find_one_and_update(
            {"_id": list_obj_id},
            {
                "$push": {
                    "items": {
                        "id": uuid4().hex,
                        "label": label,
                        "checked": False,
                    }
                }
            },
            session=session,
            return_document=ReturnDocument.AFTER,
        )
        if result:
            return ToDoList.from_doc(result)
        return None


    # async def create_item(
    #     self,
    #     id: str | ObjectId,
    #     label: str,
    #     session=None,
    # ) -> ToDoList | None:
    #     result = await self._todo_collection.find_one_and_update(
    #         {"_id": ObjectId(id)},
    #         {
    #             "$push": {
    #                 "items": {
    #                     "id": uuid4().hex,
    #                     "label": label,
    #                     "checked": False,
    #                 }
    #             }
    #         },
    #         session=session,
    #         return_document=ReturnDocument.AFTER,
    #     )
    #     if result:
    #         return ToDoList.from_doc(result)

    async def set_checked_state(
        self,
        doc_id: str | ObjectId,
        item_id: str,
        checked_state: bool,
        session=None,
    ) -> ToDoList | None:
        result = await self._todo_collection.find_one_and_update(
            {"_id": ObjectId(doc_id), "items.id": item_id},
            {"$set": {"items.$.checked": checked_state}},
            session=session,
            return_document=ReturnDocument.AFTER,
        )
        if result:
            return ToDoList.from_doc(result)
    
    async def delete_item(
        self,
        wallet_address: str,
        doc_id: str | ObjectId,
        item_id: str,
        session=None,
    ) -> ToDoList | None:
    # Check user owns this list
        user = await self._users_collection.find_one(
            {"_id": wallet_address, "lists": ObjectId(doc_id)},
            session=session
        )
        if not user:
            return None

        result = await self._todo_collection.find_one_and_update(
            {"_id": ObjectId(doc_id)},
            {"$pull": {"items": {"id": item_id}}},
            session=session,
            return_document=ReturnDocument.AFTER,
        )
        if result:
            return ToDoList.from_doc(result)

    # async def delete_item(
    #     self,
    #     doc_id: str | ObjectId,
    #     item_id: str,
    #     session=None,
    # ) -> ToDoList | None:
    #     result = await self._todo_collection.find_one_and_update(
    #         {"_id": ObjectId(doc_id)},
    #         {"$pull": {"items": {"id": item_id}}},
    #         session=session,
    #         return_document=ReturnDocument.AFTER,
    #     )
    #     if result:
    #         return ToDoList.from_doc(result)

    async def add_suggestions(
        self, list_id: str, suggestion: str, session=None
    ) -> ToDoList:

        result = await self._todo_collection.find_one_and_update(
            {"_id": ObjectId(list_id)},
            {"$set": {"suggestion": suggestion}},
            session=session,
            return_document=ReturnDocument.AFTER,
        )

        if result:
            return ToDoList.from_doc(result)
