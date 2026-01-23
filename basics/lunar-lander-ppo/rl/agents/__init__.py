from rl.agents.a2c import A2CAgent
from rl.agents.gae import GAEAgent
from rl.agents.reinforce import ReinforceAgent
from rl.agents.reinforce_baseline import ReinforceBaselineAgent

AGENTS = {
    "reinforce": ReinforceAgent,
    "reinforce_baseline": ReinforceBaselineAgent,
    "a2c": A2CAgent,
    "gae": GAEAgent,
}
