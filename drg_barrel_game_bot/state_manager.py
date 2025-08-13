import time

class StateManager:
    '''Handles program states and logs status changes.'''
    def __init__(self) -> None:
        self._state = "On Startup"
        self._state_start_time = time.perf_counter()
        self._last_displayed_state = self._state
        print(f"Initialized StateManager with state {self._state}")

    def state_duration(self) -> float:
        return round(time.perf_counter() - self._state_start_time, 2)

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, new_state: str) -> None:
        if new_state != self._state:

            previous_state = self._state
            duration = self.state_duration()
            print(f"Status '{previous_state}' ended after {duration}s")

            self._state = new_state
            self._state_start_time = time.perf_counter()

            print(f"New Status: '{self._state}'")
            self._last_displayed_state = self._state

    def display_state(self) -> None:
        print(f"Current State: {self._state} (for {self.state_duration()}s)")
