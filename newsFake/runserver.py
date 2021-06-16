"""
This script runs the newsFake application using a development server.
"""

from os import environ
from newsFake import app
import random
import threading
import webbrowser

if __name__ == '__main__':
  
   port = 5000 
   url = "http://127.0.0.1:%s/".format(port)
   app.run(port=port, debug=False)
  # app.run()
   #HOST = environ.get('SERVER_HOST', 'localhost')
    #try:
    # PORT = int(environ.get('SERVER_PORT', '5555'))
    #except ValueError:
    #  PORT = 5555
    #app.run(HOST, PORT)
   #app.run(debug=False)
 
   