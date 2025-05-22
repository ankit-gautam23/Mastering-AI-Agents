import paho.mqtt.client as mqtt
from typing import Dict, Callable, Optional
import json
import logging
from threading import Lock

class MQTTClient:
    def __init__(self, host: str = "localhost", port: int = 1883):
        self.client = mqtt.Client()
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        self.handlers: Dict[str, Callable] = {}
        self.lock = Lock()
        self._setup_client()

    def _setup_client(self) -> None:
        """
        Set up MQTT client with callbacks.
        """
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc) -> None:
        """
        Handle connection event.
        """
        if rc == 0:
            self.logger.info("Connected to MQTT broker")
        else:
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")

    def _on_message(self, client, userdata, msg) -> None:
        """
        Handle incoming message.
        """
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            self._handle_message(topic, payload)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        """
        Handle disconnection event.
        """
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection: {rc}")

    def _handle_message(self, topic: str, payload: Dict) -> None:
        """
        Handle message by calling appropriate handler.
        """
        with self.lock:
            if topic in self.handlers:
                try:
                    self.handlers[topic](payload)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")

    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.connect(self.host, self.port)
            self.client.loop_start()
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self) -> None:
        """
        Disconnect from MQTT broker.
        """
        self.client.loop_stop()
        self.client.disconnect()

    def subscribe(self, topic: str, handler: Callable) -> bool:
        """
        Subscribe to topic with handler.
        
        Args:
            topic: Topic to subscribe to
            handler: Message handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                self.handlers[topic] = handler
            self.client.subscribe(topic)
            return True
        except Exception as e:
            self.logger.error(f"Failed to subscribe: {e}")
            return False

    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from topic.
        
        Args:
            topic: Topic to unsubscribe from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                if topic in self.handlers:
                    del self.handlers[topic]
            self.client.unsubscribe(topic)
            return True
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe: {e}")
            return False

    def publish(self, topic: str, payload: Dict) -> bool:
        """
        Publish message to topic.
        
        Args:
            topic: Topic to publish to
            payload: Message payload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            message = json.dumps(payload)
            self.client.publish(topic, message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish: {e}")
            return False

    def get_client(self) -> mqtt.Client:
        """
        Get MQTT client instance.
        
        Returns:
            MQTT client instance
        """
        return self.client 