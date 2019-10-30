"""The template of the main script of the machine learning process
"""

import pickle
import numpy as np
import games.arkanoid.communication as comm
from games.arkanoid.communication import ( \
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)

def ml_loop():

    filename = "D:\\MLGame-master\\knn.sav"
    model = pickle.load(open(filename, 'rb')) 
    ball_position_history = []
    comm.ml_ready()

    while True:
            scene_info = comm.get_scene_info()
            ball_position = scene_info.ball

            if len(ball_position_history) != 0:
                inp_temp = np.array([ball_position_history[0], ball_position_history[1], ball_position[0], ball_position[1], scene_info.platform[0]])
                input = inp_temp[np.newaxis, :]
                
                if scene_info.status == GameStatus.GAME_OVER or \
                    scene_info.status == GameStatus.GAME_PASS:
                    comm.ml_ready()
                    continue
                move = model.predict(input)

                if (move < 0):
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                elif (move > 0):
                    comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                else:
                    comm.send_instruction(scene_info.frame, PlatformAction.NONE)    
            ball_position_history = ball_position