from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()

import asyncio

browser = Browser()

agent = Agent(
	task="find the price of the new apple iphone 16e",
	llm=ChatOpenAI(model='gpt-4o', temperature=0),
	browser=browser,
)


async def main():
	await agent.run()
	input('Press Enter to close the browser...')
	await browser.close()


if __name__ == '__main__':
	asyncio.run(main())