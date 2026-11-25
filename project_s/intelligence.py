"""Consumer-specific learned intent classification; no execution authority."""
from functools import lru_cache
from continuum.teacherlab import train_nb,predict

CURRICULUM={
 'calculator':('calculate arithmetic expressions','compute sums and products','evaluate numerical equations','math tool for addition subtraction division','perform decimal arithmetic'),
 'website':('publish a business homepage','design a responsive web page','show company services and contact information','build a site for a studio','create an online business brochure'),
 'inventory':('track warehouse stock levels','manage product quantities','register goods and adjust stock','maintain a catalogue of merchandise','record items in a stock database'),
}
@lru_cache(maxsize=1)
def model():
    return train_nb({'label':label,'teacher':example,'student':''}
                    for label,examples in CURRICULUM.items() for example in examples)

def interpret(requirement):
    result=predict(model(),requirement)
    return {'capability':result['label'],'score':result['confidence'],
            'mechanism':'teacherlab multinomial naive bayes',
            'training_examples':sum(map(len,CURRICULUM.values())),
            'authority':'advisory only','calibrated':False}
