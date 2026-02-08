import time
import functools
from typing import Callable, Any, Type
from logger import logger

def retry(max_retries: int = 3, delay: int = 2, exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to retry a function call upon specific exceptions.

    Args:
        max_retries (int): Maximum number of retries allowed.
        delay (int): Delay in seconds between retries.
        exceptions (tuple): Tuple of exception types to catch and retry on.

    Returns:
        Callable: The decorated function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt}/{max_retries} failed for '{func.__name__}': {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            
            logger.error(f"Function '{func.__name__}' failed after {max_retries} attempts.")
            raise last_exception
        return wrapper
    return decorator