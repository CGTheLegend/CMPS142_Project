import tensorflow as tf

def findtargets(sess, gross, budget, targets):


    # generate targets values (gross>budget)
    for i in range(5043):
        if sess.run(gross[i]) > sess.run(budget[i]):
            sess.run(targets[i].assign(1))

    return targets
