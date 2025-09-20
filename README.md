# Navigating the Repository
All the relevant files can be found in the Fritz folder:
  **listener1.py** - The Listener 1 script that currently runs indefinitely and interacts with events.json to save events as outlined below
  **listener2.py** - The Listener 2 script that currently runs one time and interacts with events.json to retrieve events as outlined below and then cross references them with sources from Fritz
  **utils.py** - The script where all the functions authored by Natalya Pletskova exist
  **LSTM_model_production.h5** - The machine learning prediction model created by Natalya Pletskova
  **feature_scaler_O4.joblib** - The feature scaler file currently in use with the model
  **target_scaler_O4.joblib** - The target scaler file currently in use with the model
  **requirements.txt** - A list of all the required libraries to have installed before running the code
  **events.json** - A database where the events from Listener 1 are stored and then accessed by Listener 2
  **source_ids.txt** - A database containing every source id that has been processed by Listener 2
  **S\*.png** - An image of the plot for event S\*'s prediction
  **ZTF\*.png** - An image of the plot for source ZTF\*'s photometry data mapped onto one of its matching events
  ***skymaps*** - A directory filled with all the skymaps that get retrieved by Listener 2

# Process
An 'event' is found through a gravitational wave trigger and 'sources' are found through measuring light emissions at specific points in the sky; we want to see which light sources belong to which gravitational wave events. This repository goes through a process to create and post plots for new sources in the Fritz EM+GW group. This is the process:
  1. Listener 1 (listener1.py) subscribes to a Kafka consumer and indefinitely queries for new GCN Events in the consumer. The GCN Events as an object correspond to Gravitational Wave (GW) triggers found in the sky. The listener ignores testing events (events starting with 'M').
  2. Listener 1 takes the new Event and receives its parameters. The Event is ran through Natalya Pletskova's Regressor Prediction model (LSTM_model_production.h5) to create an expected prediction for what the Event's Electromagnetic (EM) counterpart would be if it were a kilonova.
  3. Listener 1 saves the Event and its prediction data to a locally stored events database (events.json)
  4. Listener 2 (listener2.py) runs independently of Listener 1. Listener 2 queries the Fritz SkyPortal api using the built in pagination system to retrieve sources from newest to oldest. It stops querying when it has found a source older than 10 days or a source that has already been processed (logged in source_ids.txt).
  5. Listener 2 iterates through every source it's queried and checks to see if it matches with any of the Events in the events database. A match is determined if the source's time is within 3 days before or 7 days after the Event's alert time, as well as if the source's ra/dec coordinate location is within the 90% confidence range of the Event's skymap. Event skymaps (currently only bayestar fits skymaps work) are downloaded into a skymaps repository. When the source gets processed in this manner, it gets added to a source id database (source_ids.txt) so that it doesn't get processed again in a future query.
  6. If the source matches with an Event, a plot is created using the prediction data for the Event. The source's photometry is then placed on the Event's plot. This creates a plot where you can compare the expected light emission of the event as if it were a kilonova, with its actual recorded light emission. The closer the photometry points are to the curve of the prediction, the more likely it is that it is a kilonova, however at this point in the process, we will probably only see points near the alert time.
  7. Once the plot is made, it gets posted as a comment on Fritz under the source. When looking on the Fritz page's comments, click the 'Include Bots' checkbox to see the comment. The plot is posted as a png.

# Alternate Process (Not Currently Implemented)
An alternate process similar to the process mentioned above will be used to query candidates instead of sources (candidates are EM sources that haven't been saved to a group yet) that pass the EM+GW group filter. This process will instead work with the GW+EM Sub-Threshold group (still using the EM+GW filter) using a modified version of Listener 2. As mentioned already, Listener 2 will query candidates and then match them with Events from the events database just like in the regular Process. However, after step 5 of the regular Process, the Alternate Process takes a different approach:
  6. If the candidate matches with an Event, a forced photometry is run on the candidate to get more accurate up to date data.
  7. The forced photometry data points are compared with the Event prediction to create a reduced chi squared value. If the reduced chi squared value is between 0.1 and 10, we would consider the candidate's photometry to be close enough to indicate the possibility of being a kilonova.
  8. If the forced photometry is "close enough" to the Event prediction, we save the candidate as a source to GW+EM Sub-Threshold and then send a Slack alert to the members of the group.
  9. At this point we resume steps 6 and 7 of the regular Process to create a plot and post it as a comment on the source's page on Fritz.

  # Listener 1
  The script for this listener has instructions to follow before you run. Namely, setting up credentials in environment variables to be able to access the Kafka consumer. There also exists a debug portion currently commented out that can be used to run the code for a specific event. Currently, Listener 1 runs indefinitely using a while True loop in the script.

  # Listener 2
  The script for this listener also requires you to set credentials for accessing the SkyPortal api. Currently, Listener 2 runs one time, however it has code commented out to be able to run indefinitely, but execute at a specified time of day.

  # Events database
  The events.json file saves the events with this file format:
  {
    superevent_id: {
      "time_single": Python list
      "mean_preds_inverted": Python list of Python lists
      "uncertainty_reshaped": Python list of Python lists
      "time": string
      "alert_type": string
    },
    ...
  }

  # Source ID database
  The source_ids.txt file saves the source ids with this file format:
  source1id
  source2id
  source3id
  ...
