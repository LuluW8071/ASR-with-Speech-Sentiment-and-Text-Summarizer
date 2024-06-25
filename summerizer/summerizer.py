#this is extractive summarizer
from summarizer import Summarizer

body = """As humans are progressing as a human race into the future, the true essence of humanity is being corrupted slowly. It is essential to remember that the acts of humanity must not have any kind of personal gain behind them like fame, money or power.
The world we live in today is divided by borders but the reach we can have is limitless. We are lucky enough to have the freedom to travel anywhere and experience anything we wish for. A lot of nations fight constantly to acquire land which results in the loss of many innocent lives.
Similarly, other humanitarian crisis like the ones in Yemen, Syria, Myanmar and more costs the lives of more than millions of people. The situation is not resolving anytime soon, thus we need humanity for this.
Most importantly, humanity does not just limit to humans but also caring for the environment and every living being. We must all come together to show true humanity and help out other humans, animals and our environment to heal and prosper."""


#b#ody2 = 'Something else you want to summarize with BERT'
model = Summarizer()
model(body)
#model(body2)
