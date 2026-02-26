                                                                                                   
  # Watch one game (game narrative + action choices printed)
  python train.py --observe checkpoints/robomage_final.zip
                                                                                                   
  # Win rate vs random over 100 games
  python train.py --baseline checkpoints/robomage_final.zip                                        
                                                                                                 
  # Fewer games for a quick check
  python train.py --baseline checkpoints/robomage_25000_steps.zip --baseline-games 20

  A well-trained model should push the baseline win rate well above 50%. If it's still near 50%
  after a full training run, that's a sign the reward signal (win/loss only) is too sparse and
  intermediate rewards for things like dealing damage would help.