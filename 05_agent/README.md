# Module 5: agent post-training

1. Create small repositories with `task.json` and tests.
2. Generate teacher trajectories with `src.agent.run_agent` / `src.agent.generate_trajectories`.
3. Replay them from a clean repo and keep only reproducible successes.
4. Convert successful trajectories to SFT records and train a second adapter on top of `python_dsa_v1`.
5. Evaluate on unseen repositories and report task success, valid tool calls, test pass rate, recovery and excessive edits.
