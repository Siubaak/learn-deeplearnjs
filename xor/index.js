const dl = require('deeplearn')

const graph = new dl.Graph()
const x = graph.placeholder('x', [2])
const h1 = graph.layers.dense('h1', x, 3, x => graph.sigmoid(x), true)
const y = graph.layers.dense('y', h1, 1, x => graph.sigmoid(x), true)
const yLabel = graph.placeholder('y', [1])

const cost = graph.meanSquaredCost(yLabel, y)

const math = new dl.NDArrayMathCPU()
const session = new dl.Session(graph, math)

const xs = [
  dl.Array1D.new([0, 0]),
  dl.Array1D.new([0, 1]),
  dl.Array1D.new([1, 0]),
  dl.Array1D.new([1, 1])
]
const ys = [
  dl.Array1D.new([0]),
  dl.Array1D.new([1]),
  dl.Array1D.new([1]),
  dl.Array1D.new([0])
]

const shuffledInputProviderBuilder = new dl.InCPUMemoryShuffledInputProviderBuilder([ xs, ys ])
const [ xProvider, yProvider ] = shuffledInputProviderBuilder.getInputProviders()

const NUM_BATCHES = 20000
const BATCH_SIZE = xs.length
const LEARNING_RATE = 0.3
const optimizer = new dl.SGDOptimizer(LEARNING_RATE)
for (let i = 0; i < NUM_BATCHES; i++) {
  const costValue = session.train(
      cost,
      [
        {
          tensor: x,
          data: xProvider
        },
        {
          tensor: yLabel,
          data: yProvider
        }
      ],
      BATCH_SIZE, optimizer, dl.CostReduction.MEAN)
      if (i % 1000 === 0) console.log(costValue.getValues())
}

result = session.eval(y, [
  {
    tensor: x,
    data: dl.Array1D.new([0, 1])
  }
])
console.log(result.getValues())

