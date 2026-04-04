from dotenv import load_dotenv
from calculationqueue import CalculationQueue
import sys

load_dotenv()

queue : CalculationQueue = CalculationQueue(sys.argv[1])
queue.run()