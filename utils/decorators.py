import time
import functools
from typing import Callable, Any, Type, Tuple
from logger import logger

class ExecutionDecorators:
    """
    Utility class containing common decorators for execution control.
    """

    @staticmethod
    def retry(max_retries: int = 3, delay: int = 2, exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable:
        """
        A decorator that retries a function call upon specific exceptions.

        Args:
            max_retries (int): Maximum number of retry attempts. Defaults to 3.
            delay (int): Delay in seconds between retries. Defaults to 2.
            exceptions (Tuple[Type[Exception], ...]): Exception types to catch and retry. 
                Defaults to (Exception,).

        Returns:
            Callable: The decorated function.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception = None
                for attempt in range(1, max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        logger.warning(
                            f"Attempt {attempt}/{max_retries} failed for '{func.__name__}': {e}. "
                            f"Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                
                logger.error(f"Function '{func.__name__}' failed after {max_retries} attempts.")
                raise last_exception
            return wrapper
        return decorator
