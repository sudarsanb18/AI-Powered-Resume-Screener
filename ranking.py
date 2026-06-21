def rank_candidates(candidates):

    ranked=sorted(

        candidates,

        key=lambda x:x["score"],

        reverse=True

    )

    return ranked