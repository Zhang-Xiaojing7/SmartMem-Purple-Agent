TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "interact_with_environment",
            "description": "Interact with smart home devices to read their status or update their state. If device_id is set to environment, only the read operation can be selected, which will return the status of all devices in the current environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "The specific device identifier to interact with.",
                        "enum": [
                            "living_room_light",
                            "living_room_color",
                            "bedroom_light",
                            "bedroom_color",
                            "ac",             # ac_power
                            "ac_temperature",
                            "fan_speed",
                            "music_volume",
                            "front_door_lock",
                            "kitchen_light"
                        ]
                    },
                    "action": {
                        "type": "string",
                        "enum": ["read", "update"],
                        "description": "Whether to 'read' the current state or 'update' to a new state."
                    },
                    "value": {
                        "type": "string",
                        "description": "The target value for 'update' action. Ignored if action is 'read'.\n"
                                       "- Lights/AC Power: 'on', 'off'\n"
                                       "- Colors: 'white', 'red', 'blue', 'warm'\n"
                                       "- Temperature: Integer string '16' to '30'\n"
                                       "- Fan Speed: 'off', 'low', 'medium', 'high'\n"
                                       "- Volume: Integer string '0' to '10'\n"
                                       "- Lock: 'locked', 'unlocked'"
                    }
                },
                "required": ["device_id", "action"]
            }
        }
    }
]