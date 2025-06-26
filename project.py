import os
import logging
from dotenv import load_dotenv
from playwright.sync_api import (
    sync_playwright,
    Page,
    Locator,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)
from langchain_openai import ChatOpenAI

# Import necessary components for inspecting intermediate steps if needed
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)  # Import ToolMessage
from pydantic.v1 import BaseModel, Field  # Keep v1 for LangChain compatibility
from langchain.tools import BaseTool
from typing import Type, Optional, Dict, List, ClassVar
from bs4 import BeautifulSoup
import time
import re
import argparse
import json
from browser_use import Agent, Browser
import asyncio

# --- Configuration & Setup ---
load_dotenv()
browser = Browser()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# logging.getLogger("playwright").setLevel(logging.DEBUG)

# --- Load configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

# --- Retry Configuration ---
MAX_RETRIES = 2
INITIAL_RETRY_DELAY = 2.0


# --- Shared Browser State Wrapper ---
class BrowserState:
    def __init__(self, page: Page):
        self.page: Page = page
        self.buid_map: Dict[int, Locator] = {}

    def clear_buid_map(self):
        logging.debug("Clearing BUID map.")
        self.buid_map.clear()

    def get_page(self) -> Page:
        if self.page.is_closed():
            raise PlaywrightError("Browser page is closed!")
        return self.page


# --- Playwright Browser Tools ---
class PlaywrightBaseTool(BaseTool):
    browser_state: BrowserState = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True


# --- Tool Input Schemas ---
# (Keep schemas including AskHumanInput)
class NavigateURLInput(BaseModel):
    url: str = Field(
        description="The URL to navigate to in the current browser tab. Should be a fully qualified URL (e.g., https://www.google.com, https://www.example.com)."
    )


class TypeTextInput(BaseModel):
    buid: int = Field(
        description="The browser-assigned ID (buid) of the input element to type into."
    )
    text: str = Field(description="The literal text to type into the element.")
    press_enter: bool = Field(
        default=False, description="Whether to press Enter after typing."
    )


class ClickElementInput(BaseModel):
    buid: int = Field(
        description="The browser-assigned ID (buid) of the element to click."
    )


class GetContentInput(BaseModel):
    selector: Optional[str] = Field(
        default="body",
        description="Optional CSS selector to focus the content retrieval area (e.g., 'main', '#search', 'details-menu[role=menu]'). Defaults to 'body'.",
    )


class WaitForElementInput(BaseModel):
    selector: str = Field(description="CSS selector of the element to wait for.")
    timeout: Optional[int] = Field(
        default=10000,
        description="Maximum time in milliseconds to wait for the element to be visible.",
    )


class GetTextInput(BaseModel):
    selector: str = Field(
        description="CSS selector of the element from which to extract text content."
    )


class AskHumanInput(BaseModel):
    question: str = Field(
        description="The specific question or instruction for the human user (e.g., 'I seem to be blocked by a CAPTCHA. Please solve it and press Enter.')."
    )


# --- Tool Implementations ---
# (Keep NavigateTool, GetContentTool, TypeTextTool, ClickElementTool, WaitForElementTool, GetTextFromSelectorTool implementations exactly as before - WITHOUT internal intervention logic)
# ... (Include the full, unchanged code for these six tool classes here) ...
class NavigateTool(PlaywrightBaseTool):
    name: str = "navigate_url"
    description: str = (
        "Navigates the current browser tab to the specified URL. Invalidates all previous buids. Retries on failure."
    )
    args_schema: Type[BaseModel] = NavigateURLInput

    def _run(self, url: str) -> str:
        self.browser_state.clear_buid_map()
        page = self.browser_state.get_page()
        if not re.match(r"^https?://", url):
            url = f"https://{url}"
            logging.warning(f"URL '{url}' was missing scheme, prefixed with https://")
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(
                    f"[Attempt {attempt + 1}/{MAX_RETRIES}] Navigating tab to: {url}"
                )
                response = page.goto(url, wait_until="load", timeout=40000)
                status = response.status if response else "unknown"
                current_url = page.url
                logging.info(
                    f"Navigation complete. Status: {status}. Current URL: {current_url}"
                )
                time.sleep(2.5)
                return f"Successfully navigated tab to {url}. Page status: {status}. Current URL is {current_url}. IMPORTANT: All previous buids are now invalid. Use get_page_content to get new ones."
            except (PlaywrightTimeoutError, PlaywrightError) as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Navigation error for {url}. Error: {e}"
                )
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (2**attempt)
                    logging.info(f"Retrying navigation in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(
                        f"All {MAX_RETRIES} navigation attempts failed for {url}. Last error: {last_exception}"
                    )
                    return f"Error: FailedToNavigate - URL: {url}, Attempts: {MAX_RETRIES}, LastError: {str(last_exception)[:200]}"
            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Unexpected error during navigation to {url}. Error: {e}",
                    exc_info=True,
                )
                return f"Error: UnexpectedNavigationError - URL: {url}, Error: {str(e)[:200]}"
        return f"Error: FailedToNavigate - URL: {url}, Attempts: {MAX_RETRIES}, Reason: Unknown failure after loop."


class GetContentTool(PlaywrightBaseTool):
    name: str = "get_page_content"
    description: str = (
        "Retrieves the current page's simplified content, assigning unique IDs ([buid=N]) to interactable elements (links, buttons, inputs, textareas, selects) within the specified CSS selector area (defaults to 'body'). Use this to understand the page and get buids for interaction tools. Invalidates previous buids."
    )
    args_schema: Type[BaseModel] = GetContentInput
    INTERACTABLE_SELECTOR: ClassVar[str] = (
        "a, button, input:not([type=hidden]), textarea, select"
    )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _extract_and_annotate(self, element: Locator, buid: int) -> Optional[str]:
        try:
            tag = element.evaluate("el => el.tagName.toLowerCase()")
            text = ""
            value = ""
            placeholder = ""
            aria_label = element.get_attribute("aria-label") or ""
            data_testid = element.get_attribute("data-testid") or ""
            title = element.get_attribute("title") or ""
            if tag in ["a", "button"]:
                text = element.text_content(timeout=1000) or ""
            elif tag in ["input", "textarea"]:
                value = element.input_value(timeout=1000) or ""
                placeholder = element.get_attribute("placeholder") or ""
                text = aria_label if aria_label else placeholder
            elif tag == "select":
                selected_option = element.locator("option:checked")
                if selected_option.count() > 0:
                    text = selected_option.first.text_content(timeout=1000) or ""
                else:
                    text = (
                        aria_label
                        if aria_label
                        else (
                            element.locator("option").first.text_content(timeout=1000)
                            or ""
                        )
                    )
            text = self._clean_text(text)
            value = self._clean_text(value)
            aria_label = self._clean_text(aria_label)
            data_testid = self._clean_text(data_testid)
            title = self._clean_text(title)
            representation = f"<{tag}"
            if data_testid:
                representation += f" data-testid='{data_testid}'"
            if aria_label:
                representation += f" aria-label='{aria_label}'"
            if title and title != text and title != aria_label:
                representation += f" title='{title}'"
            if placeholder:
                representation += f" placeholder='{placeholder}'"
            if value:
                representation += f" value='{value}'"
            representation += f"> {text} </{tag}> [buid={buid}]"
            return representation
        except Exception as e:
            logging.warning(f"Could not extract/annotate buid {buid}. Error: {e}")
            return None

    def _run(self, selector: str = "body") -> str:
        page = self.browser_state.get_page()
        buid_map = self.browser_state.buid_map
        buid_map.clear()
        logging.info(f"Getting annotated content within selector: '{selector}'")
        try:
            if selector != "body":
                time.sleep(0.5)
            page.wait_for_selector(selector, state="attached", timeout=15000)
            base_element = page.locator(selector).first
            interactable_elements = base_element.locator(self.INTERACTABLE_SELECTOR)
            count = interactable_elements.count()
            log_prefix = f"[{selector}] " if selector != "body" else ""
            logging.info(f"{log_prefix}Found {count} potential interactable elements.")
            visible_elements_data = []
            buid_counter = 1
            max_elements_to_process = 200
            processed_count = 0
            all_locators = interactable_elements.all()
            if len(all_locators) > max_elements_to_process:
                logging.warning(
                    f"{log_prefix}Found {len(all_locators)} interactable elements, limiting processing to first {max_elements_to_process}."
                )
                all_locators = all_locators[:max_elements_to_process]
            for i, element_locator in enumerate(all_locators):
                current_buid = -1
                try:
                    visibility_timeout = 750 if selector != "body" else 500
                    if element_locator.is_visible(timeout=visibility_timeout):
                        current_buid = buid_counter
                        buid_map[current_buid] = element_locator
                        annotated_string = self._extract_and_annotate(
                            element_locator, current_buid
                        )
                        if annotated_string:
                            visible_elements_data.append(annotated_string)
                        else:
                            del buid_map[current_buid]
                        buid_counter += 1
                except Exception as e:
                    logging.debug(f"{log_prefix}Skipping element {i} due to error: {e}")
                    if current_buid != -1 and current_buid in buid_map:
                        del buid_map[current_buid]
                processed_count += 1
            logging.info(
                f"{log_prefix}Generated {len(buid_map)} buids for visible interactable elements."
            )
            simplified_content = "\n".join(visible_elements_data)
            # Add back the CAPTCHA detection hint for the LLM
            if (
                selector == "body"
                and len(buid_map) <= 2
                and any(
                    "captcha" in d.lower() or "why did this happen" in d.lower()
                    for d in visible_elements_data
                )
            ):
                logging.warning(
                    "Detected possible CAPTCHA page based on limited content."
                )
                simplified_content += "\n\n[INFO] Page content is very limited, potentially indicating a CAPTCHA or block page."

            if not simplified_content:
                return f"No visible interactable elements found within selector '{selector}'."
            max_len = 8000
            if len(simplified_content) > max_len:
                simplified_content = (
                    simplified_content[:max_len] + "\n... [Content Truncated]"
                )
            return simplified_content
        except Exception as e:
            logging.error(
                f"Error during get_page_content for selector '{selector}': {e}",
                exc_info=True,
            )
            buid_map.clear()
            return f"Error: FailedToGetAnnotatedContent - Selector: {selector}, Error: {str(e)[:200]}"


class TypeTextTool(PlaywrightBaseTool):
    name: str = "type_text"
    description: str = (
        "Types the given literal text into the element specified by its browser-assigned ID (buid). Invalidates all buids. Retries on failure."
    )
    args_schema: Type[BaseModel] = TypeTextInput

    def _run(self, buid: int, text: str, press_enter: bool = False) -> str:
        page = self.browser_state.get_page()
        buid_map = self.browser_state.buid_map
        logging.info(f"Attempting to type into element with buid={buid}")
        element_locator = buid_map.get(buid)
        if not element_locator:
            logging.error(
                f"BUID {buid} not found in the current map. It might be stale. Use get_page_content again."
            )
            return f"Error: InvalidBUID - BUID {buid} is from a previous page state. You MUST call get_page_content NOW to get the current valid BUIDs before trying any other action."
        actual_text = text
        log_text = text[:50] + ("..." if len(text) > 50 else "")
        if "password" in log_text.lower():
            log_text = "[PASSWORD]"
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(
                    f"[Attempt {attempt + 1}/{MAX_RETRIES}] Typing into buid={buid} (Press Enter: {press_enter}) Text: '{log_text}'"
                )
                logging.debug(f"Scrolling buid={buid} into view if needed...")
                element_locator.scroll_into_view_if_needed(timeout=5000)
                logging.debug(f"Focusing buid={buid}...")
                element_locator.focus(timeout=5000)
                time.sleep(0.2)
                logging.debug(f"Filling buid={buid}...")
                element_locator.clear(timeout=5000)
                element_locator.fill(actual_text, timeout=20000)
                if press_enter:
                    logging.debug(f"Pressing Enter on buid={buid}...")
                    element_locator.press("Enter", timeout=5000)
                    logging.info("Pressed Enter.")
                    try:
                        logging.debug("Waiting for network idle after Enter...")
                        page.wait_for_load_state("networkidle", timeout=15000)
                    except PlaywrightTimeoutError:
                        logging.warning(
                            "Timeout waiting for network idle after pressing Enter, proceeding."
                        )
                    time.sleep(1.5)
                self.browser_state.clear_buid_map()
                logging.info(
                    f"Successfully typed text '{log_text}' into buid={buid}. BUIDs invalidated."
                )
                return f"Successfully typed text '{log_text}' into buid={buid}. IMPORTANT: All buids are now invalid. Use get_page_content to get new ones."
            except (PlaywrightTimeoutError, PlaywrightError) as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Playwright error typing into buid={buid}. Error: {e}"
                )
                last_exception = e
                if (
                    "element is not attached" in str(e).lower()
                    or "frame was detached" in str(e).lower()
                ):
                    logging.error(
                        f"Element for buid={buid} became detached. Aborting retries."
                    )
                    self.browser_state.clear_buid_map()
                    return f"Error: ElementDetached - BUID {buid} became detached. Get fresh content. BUIDs invalidated."
                if press_enter and "navigation" in str(e).lower():
                    logging.warning(
                        f"Navigation likely interrupted typing/Enter for buid={buid}. Assuming action initiated navigation."
                    )
                    self.browser_state.clear_buid_map()
                    return f"Typed text '{log_text}' into buid={buid}, pressed Enter, and navigation started. IMPORTANT: All buids are now invalid. Use get_page_content to get new ones."
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (2**attempt)
                    logging.info(f"Retrying typing in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(
                        f"All {MAX_RETRIES} typing attempts failed for buid={buid}. Last error: {last_exception}"
                    )
                    # Return normal error, let LLM decide to ask for help
                    self.browser_state.clear_buid_map()
                    return f"Error: FailedToType - BUID: {buid}, Attempts: {MAX_RETRIES}, LastError: {str(last_exception)[:200]}. BUIDs invalidated."
            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Unexpected error typing into buid={buid}. Error: {e}",
                    exc_info=True,
                )
                self.browser_state.clear_buid_map()
                return f"Error: UnexpectedTypeError - BUID: {buid}, Error: {str(e)[:200]}. BUIDs invalidated."
        self.browser_state.clear_buid_map()
        return f"Error: FailedToType - BUID: {buid}, Attempts: {MAX_RETRIES}, Reason: Unknown failure after loop. BUIDs invalidated."


class ClickElementTool(PlaywrightBaseTool):
    name: str = "click_element"
    description: str = (
        "Clicks the element specified by its browser-assigned ID (buid). Invalidates all buids. Retries on failure."
    )
    args_schema: Type[BaseModel] = ClickElementInput

    def _run(self, buid: int) -> str:
        page = self.browser_state.get_page()
        buid_map = self.browser_state.buid_map
        logging.info(f"Attempting to click element with buid={buid}")
        element_locator = buid_map.get(buid)
        if not element_locator:
            logging.error(
                f"BUID {buid} not found in the current map. It might be stale. Use get_page_content again."
            )
            return f"Error: InvalidBUID - BUID {buid} is from a previous page state. You MUST call get_page_content NOW to get the current valid BUIDs before trying any other action."
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(
                    f"[Attempt {attempt + 1}/{MAX_RETRIES}] Clicking element with buid={buid}"
                )
                logging.debug(f"Scrolling buid={buid} into view if needed...")
                element_locator.scroll_into_view_if_needed(timeout=5000)
                logging.debug(f"Performing click on buid={buid}...")
                element_locator.click(timeout=20000)
                logging.info(
                    f"Clicked element buid={buid} successfully on attempt {attempt + 1}. Waiting for potential navigation/update..."
                )
                time.sleep(1.2)
                try:
                    logging.debug(
                        "Waiting for network idle after click (and potential menu)..."
                    )
                    page.wait_for_load_state("networkidle", timeout=15000)
                except PlaywrightTimeoutError:
                    logging.warning(
                        f"Timeout waiting for network idle after clicking buid={buid}, page might be partially loaded or dynamic."
                    )
                time.sleep(1.0)
                current_url = page.url
                self.browser_state.clear_buid_map()
                logging.info(
                    f"Successfully clicked buid={buid}. BUIDs invalidated. Current URL is now {current_url}."
                )
                return f"Successfully clicked buid={buid}. Current URL is now {current_url}. IMPORTANT: All buids are now invalid. If this click opened a menu/modal, use get_page_content with a specific selector for that menu next. Otherwise, use get_page_content normally."
            except (PlaywrightTimeoutError, PlaywrightError) as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Playwright error clicking buid={buid}. Error: {e}"
                )
                last_exception = e
                if (
                    "element is not attached" in str(e).lower()
                    or "frame was detached" in str(e).lower()
                ):
                    logging.error(
                        f"Element for buid={buid} became detached. Aborting retries."
                    )
                    self.browser_state.clear_buid_map()
                    return f"Error: ElementDetached - BUID {buid} became detached. Get fresh content. BUIDs invalidated."
                if "intercepts pointer events" in str(e).lower():
                    logging.warning(
                        f"Click on buid={buid} might have been intercepted by an overlay/menu. Trying get_page_content next might reveal it."
                    )
                    self.browser_state.clear_buid_map()
                    return f"Warning: Click on buid={buid} failed, possibly intercepted by an overlay/menu. BUIDs invalidated. Use get_page_content to check the current state."
                if "navigation" in str(e).lower():
                    logging.warning(
                        f"Navigation likely interrupted click for buid={buid} on attempt {attempt + 1}. Assuming click initiated navigation."
                    )
                    current_url = page.url
                    try:
                        page.wait_for_load_state("load", timeout=7000)
                    except PlaywrightTimeoutError:
                        pass
                    time.sleep(1.5)
                    self.browser_state.clear_buid_map()
                    return f"Clicked element buid={buid} and navigation started. Current URL is now {current_url}. IMPORTANT: All buids are now invalid. Use get_page_content to get new ones."
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (2**attempt)
                    logging.info(f"Retrying click in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(
                        f"All {MAX_RETRIES} click attempts failed for buid={buid}. Last error: {last_exception}"
                    )
                    # Return normal error, let LLM decide to ask for help
                    self.browser_state.clear_buid_map()
                    return f"Error: FailedToClick - BUID: {buid}, Attempts: {MAX_RETRIES}, LastError: {str(last_exception)[:200]}. BUIDs invalidated."
            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Unexpected error clicking buid={buid}. Error: {e}",
                    exc_info=True,
                )
                self.browser_state.clear_buid_map()
                return f"Error: UnexpectedClickError - BUID: {buid}, Error: {str(e)[:200]}. BUIDs invalidated."
        self.browser_state.clear_buid_map()
        return f"Error: FailedToClick - BUID: {buid}, Attempts: {MAX_RETRIES}, Reason: Unknown failure after loop. BUIDs invalidated."


class WaitForElementTool(PlaywrightBaseTool):
    name: str = "wait_for_element_visible"
    description: str = (
        "Waits for an element specified by a CSS selector to become visible on the page. Use this AFTER clicking something that should reveal a new element (like a menu item) and BEFORE trying to get content or interact with that new element."
    )
    args_schema: Type[BaseModel] = WaitForElementInput

    def _run(self, selector: str, timeout: int = 10000) -> str:
        page = self.browser_state.get_page()
        if page.is_closed():
            return "Error: The browser page is closed. Cannot wait."
        logging.info(
            f"Waiting up to {timeout}ms for selector '{selector}' to become visible..."
        )
        try:
            page.wait_for_selector(selector, state="visible", timeout=timeout)
            logging.info(f"Element '{selector}' is now visible.")
            return f"Element '{selector}' became visible."
        except PlaywrightTimeoutError:
            logging.error(
                f"Timeout: Element '{selector}' did not become visible within {timeout}ms."
            )
            return f"Error: Timeout - Element '{selector}' did not become visible within {timeout}ms."
        except Exception as e:
            logging.error(
                f"Unexpected error waiting for selector '{selector}': {e}",
                exc_info=True,
            )
            return f"Error: UnexpectedWaitError - Selector: {selector}, Error: {str(e)[:200]}"


class GetTextFromSelectorTool(PlaywrightBaseTool):
    name: str = "get_text_from_selector"
    description: str = (
        "Extracts and returns the text content from the first element matching the given CSS selector. Use this to get specific information like prices, descriptions, titles, etc., when the text is not part of an interactable element found by get_page_content. Does NOT invalidate buids. Retries on failure."
    )
    args_schema: Type[BaseModel] = GetTextInput

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _run(self, selector: str) -> str:
        page = self.browser_state.get_page()
        if page.is_closed():
            return "Error: The browser page is closed. Cannot get text."
        logging.info(f"Attempting to get text from selector: '{selector}'")
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                logging.info(
                    f"[Attempt {attempt + 1}/{MAX_RETRIES}] Getting text from selector '{selector}'"
                )
                page.wait_for_selector(selector, state="attached", timeout=15000)
                element = page.locator(selector).first
                text_content = element.text_content(timeout=10000)
                if text_content is None:
                    logging.warning(
                        f"Element '{selector}' found, but has no text content."
                    )
                    return f"Element '{selector}' found but contains no text."
                cleaned_text = self._clean_text(text_content)
                logging.info(
                    f"Successfully got text from '{selector}': '{cleaned_text[:100]}...'"
                )
                max_len = 1000
                if len(cleaned_text) > max_len:
                    return cleaned_text[:max_len] + " [Text Truncated]"
                return cleaned_text
            except (PlaywrightTimeoutError, PlaywrightError) as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Playwright error getting text from '{selector}'. Error: {e}"
                )
                last_exception = e
                if (
                    "does not exist" in str(e).lower()
                    or "frame was detached" in str(e).lower()
                ):
                    logging.error(
                        f"Element '{selector}' likely does not exist or detached. Aborting retries."
                    )
                    return f"Error: ElementNotFound - Selector: {selector}, Error: {str(e)[:200]}"
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (2**attempt)
                    logging.info(f"Retrying get text in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logging.error(
                        f"All {MAX_RETRIES} get text attempts failed for '{selector}'. Last error: {last_exception}"
                    )
                    return f"Error: FailedToGetText - Selector: {selector}, Attempts: {MAX_RETRIES}, LastError: {str(last_exception)[:200]}."
            except Exception as e:
                logging.error(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed: Unexpected error getting text from '{selector}'. Error: {e}",
                    exc_info=True,
                )
                return f"Error: UnexpectedGetTextError - Selector: {selector}, Error: {str(e)[:200]}."
        return f"Error: FailedToGetText - Selector: {selector}, Attempts: {MAX_RETRIES}, Reason: Unknown failure after loop."


# *** Re-add AskHumanTool ***
class AskHumanTool(
    PlaywrightBaseTool
):  # Inherit from PlaywrightBaseTool to access browser_state if needed
    name: str = "ask_human_for_help"
    description: str = (
        "Use this tool ONLY when you are completely stuck due to a likely CAPTCHA or an unresolvable UI issue after multiple retries. Asks the human user to manually intervene in the browser."
    )
    args_schema: Type[BaseModel] = AskHumanInput

    def _run(self, question: str) -> str:
        """Pauses execution and asks the human user for help."""
        logging.info(f"Asking human for help: {question}")
        print("\n*****************************************************")
        print(f"ASSISTANT NEEDS HELP: {question}")
        # Ensure the browser window is easily accessible
        try:
            self.browser_state.get_page().bring_to_front()
        except Exception as e:
            logging.warning(f"Could not bring browser page to front: {e}")

        user_input = input(
            ">>> Press Enter in this console after completing the action in the browser... "
        )
        print("*****************************************************")
        logging.info("Human has intervened. Resuming agent.")
        # This tool does NOT invalidate BUIDs itself, but the human action likely did.
        # The prompt tells the agent to call get_page_content next.
        return "Human user has intervened. IMPORTANT: The page state may have changed significantly. You MUST call get_page_content NOW to understand the current state and get new BUIDs before proceeding."

    async def _arun(self, question: str) -> str:
        # Simple synchronous wrapper for async if ever needed
        return self._run(question)


# --- LangChain Agent Setup (Prompt updated to use AskHumanTool again) ---


def create_agent(browser_state: BrowserState, llm: ChatOpenAI):
    """Creates the LangChain agent with Playwright tools using BUIDs."""
    # *** FIX: Add AskHumanTool back to the list ***
    tools = [
        NavigateTool(browser_state=browser_state),
        GetContentTool(browser_state=browser_state),
        ClickElementTool(browser_state=browser_state),
        TypeTextTool(browser_state=browser_state),
        WaitForElementTool(browser_state=browser_state),
        GetTextFromSelectorTool(browser_state=browser_state),
        AskHumanTool(browser_state=browser_state),  # Add the help tool back
    ]

    # System Prompt (Instruct LLM to use AskHumanTool when stuck)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a highly capable and autonomous web browsing assistant.
Your goal is to achieve the objective stated in the user's request (`{input}`) by intelligently interacting with websites using the provided tools. You operate by interacting with elements identified by browser-assigned IDs (buids).

**STARTING STRATEGY - VERY IMPORTANT:**
1.  **Analyze User Request (`{input}`):** Determine the core goal.
2.  **Check for Explicit URL:** Is the user's request *only* asking to navigate to a specific URL and nothing else?
    *   **IF YES:** Use `navigate_url` with that exact URL as your first action.
    *   **IF NO (DEFAULT FOR ALL OTHER REQUESTS):** Your first actions MUST be to perform a web search:
        a.  Use `navigate_url` to go to `https://www.google.com`.
        b.  Use `get_page_content` to find the search bar buid.
        c.  Extract relevant keywords from `{input}`.
        d.  Use `type_text` to enter the keywords into the search bar buid and press Enter.
3.  **DO NOT navigate directly to a brand's website based on keywords.** ALWAYS start with the Google search sequence unless the request *only* contained a URL.

**After Initial Navigation/Search:** Proceed with the CORE BUID WORKFLOW below to analyze search results or interact with the target page.

**CORE BUID WORKFLOW (Use this on ANY page):**
1.  **ALWAYS Use `get_page_content` FIRST:** Before ANY interaction (`click_element`, `type_text`), call `get_page_content` to get the current page state and valid buids for *interactable* elements.
2.  **Identify Target BUID or Need for Text:** Analyze the `get_page_content` output. Find the target element's `buid` for clicking/typing, OR identify a CSS selector for specific text extraction.
3.  **Interact or Extract:**
    *   To click/type: Call `click_element(buid=N)` or `type_text(buid=N, ...)`.
    *   To extract text: Call `get_text_from_selector(selector='your-css-selector')`.
4.  **BUIDs INVALIDATED (After Actions):** After `navigate_url`, `click_element`, or `type_text` successfully executes, **ALL BUIDs become INVALID**. `get_text_from_selector` and `wait_for_element_visible` do NOT invalidate buids.
5.  **REPEAT:** If BUIDs were invalidated, immediately go back to Step 1 (`get_page_content`) to get fresh buids before your *next* interaction. If you only used `get_text_from_selector`, you can proceed with another action using the *existing* valid buids if appropriate.

**CRITICAL: Handling Menus/Modals/Overlays:**
*   **AFTER** clicking a buid expected to open a menu/modal:
*   **STEP A: Wait:** Your **VERY NEXT ACTION** MUST be `wait_for_element_visible`. Provide a CSS selector for a known item *inside* the menu.
*   **STEP B: Get Focused Content:** If wait succeeds, **NEXT ACTION** MUST be `get_page_content` targeting the menu container selector.
*   **STEP C: Interact Within Menu:** If focused get succeeds, find the needed buid *within that focused content* and interact.
*   **RECOVERY:** If Step A or B fails, call `get_page_content` on `body`, re-assess, try a *different* menu selector for Step B. **DO NOT** navigate directly as a workaround.
*   **AFTER MENU:** Call `get_page_content` (no selector) to refresh main page buids.

**HANDLING FAILURES & CAPTCHAs:**
*   Tools retry basic errors internally. If a tool returns a persistent Error (e.g., `FailedToClick`, `FailedToType`, `ElementNotFound`, `Timeout`):
    *   Analyze the error and the latest `get_page_content` output.
    *   **IF** you are stuck on a search results page (like Google) or a login page, and repeatedly fail to find or interact with expected elements after retries (e.g., `get_page_content` returns very little content or errors like "Possible CAPTCHA detected"):
    *   **THEN:** Your **ONLY** valid next action is to use the `ask_human_for_help` tool. Provide a clear `question` asking the user to resolve the block (e.g., "I suspect a CAPTCHA is blocking the page. Please solve it in the browser window and press Enter in the console."). **DO NOT try any other tool or give up.**
    *   **AFTER `ask_human_for_help` returns:** Your **VERY NEXT ACTION** MUST be `get_page_content` to understand the new page state. Then continue your task based on the new content.
    *   If not a CAPTCHA scenario, try alternative BUIDs/strategies or report failure if truly stuck.

**Login:** Only attempt login if required by the prompt AND credentials are provided in the prompt. Follow BUID workflow. Verify success by looking for positive indicators.

Available Tools:
- navigate_url(url: str)
- get_page_content(selector: str = "body")
- click_element(buid: int)
- type_text(buid: int, text: str, press_enter: bool = False)
- wait_for_element_visible(selector: str, timeout: int = 10000)
- get_text_from_selector(selector: str): Extracts text content. Does NOT invalidate buids.
- ask_human_for_help(question: str): Use ONLY when stuck due to likely CAPTCHA or unresolvable UI issue after multiple retries. MUST call get_page_content immediately after.

Think step-by-step. Describe reasoning, plan, and **target buids/selectors** before acting. **Strictly follow the Search First strategy and BUID workflow.**
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output and make sure it conforms to the BUID workflow!",
        max_iterations=35,
    )
    return agent_executor


# --- Main Execution ---
def main():
    """Launches a visible Playwright browser and runs the agent."""
    parser = argparse.ArgumentParser(
        description="Run Autonomous Web Agent (Visible Browser with BUIDs)"
    )
    parser.add_argument(
        "user_task_prompt",
        type=str,
        help="The user's task for the agent (e.g., 'Log into GitHub with user MyUser pass MyPass and find my starred repos')",
    )
    args = parser.parse_args()

    # *** Use the specified model name ***
    llm = ChatOpenAI(
        model="o3-mini", openai_api_key=OPENAI_API_KEY
    )  # Adjusted as per your last correction
    logging.info(f"Using LLM model: {llm.model_name}")

    browser = None
    p_context = None
    page = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            logging.info("Launched visible browser instance.")
            p_context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36",
                viewport={"width": 1440, "height": 900},
            )
            p_context.set_default_navigation_timeout(50000)
            p_context.set_default_timeout(30000)
            page = p_context.new_page()
            logging.info("Opened new page for agent interaction.")
            browser_state = BrowserState(page=page)
            agent_executor = create_agent(browser_state, llm)
            initial_query = args.user_task_prompt
            chat_history = []
            logging.info(f"Starting agent execution for prompt: '{initial_query}'")

            # Agent execution happens here. AskHumanTool handles pausing.
            result = agent_executor.invoke(
                {"input": initial_query, "chat_history": chat_history}
            )

            final_output = result.get("output", "No output field found.")
            logging.info("Agent execution finished.")
            logging.info(f"Final Output: {final_output}")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info("Script finished.")  # Cleanup handled by context manager


agent = Agent(
    task="find the price of an allen solly suit",
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    browser=browser,
)


async def main_run():
    await agent.run()
    input("Press Enter to close the browser...")
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main_run())
