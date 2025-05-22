from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
import uuid

@dataclass
class User:
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: str = "user"
    preferences: Dict = field(default_factory=dict)

class UserManager:
    def __init__(self):
        self.users = {}

    def create_user(self, username: str, email: str, role: str = "user") -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            role: User role
            
        Returns:
            Created User object
        """
        # TODO: Implement user creation
        # 1. Validate input
        # 2. Create user object
        # 3. Store user
        pass

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object if found, None otherwise
        """
        # TODO: Implement user retrieval
        # 1. Check if user exists
        # 2. Return user object
        pass

    def update_user(self, user_id: str, updates: Dict) -> Optional[User]:
        """
        Update user information.
        
        Args:
            user_id: User ID
            updates: Dictionary of updates to apply
            
        Returns:
            Updated User object if successful, None otherwise
        """
        # TODO: Implement user update
        # 1. Validate user exists
        # 2. Apply updates
        # 3. Store changes
        pass

    def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        # TODO: Implement user deletion
        # 1. Validate user exists
        # 2. Remove user
        pass

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User object if authenticated, None otherwise
        """
        # TODO: Implement user authentication
        # 1. Validate credentials
        # 2. Return user object
        pass 