class RetentionPolicyException(Exception):
    def __init__(self):
        message = '\nretention_policy must have one of the following values:\n'
        message += '\t- NR: Never retain\n'
        message += '\t- AR: Always retain\n'
        message += '\t- DF: Different Class retention\n'
        message += '\t- DD: Degree of Disagreement\n'
        message += 'Check the documentation for further details.'
        super().__init__(message)


class VotingPolicyException(Exception):
    def __init__(self):
        message = '\nvoting_policy must have one of the following values:\n'
        message += '\t- MVS: Most Voted Solution\n'
        message += '\t- MP: Modified Plurality\n'
        super().__init__(message)


class TestMethodException(Exception):
    def __init__(self):
        message = '\ntest_method must have one of the following values:\n'
        message += '\t- anova\n'
        message += '\t- wilcoxon\n'
        message += '\t- ttest\n'
        super().__init__(message)
