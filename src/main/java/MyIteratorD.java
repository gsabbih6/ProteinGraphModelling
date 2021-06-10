import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

public class MyIteratorD implements MultiDataSetIterator {
    private Iterator<MultiDataSet> iterator;
    private List<MultiDataSet> list;
    private final int batchSize;
    private final LinkedList<MultiDataSet> queued;
    private MultiDataSetPreProcessor preProcessor;

    public MyIteratorD(List<MultiDataSet> list, int batchSize) {
        this.list = list;
        this.iterator = list.iterator();
        this.batchSize = batchSize;
        this.queued = new LinkedList();
    }

    public boolean hasNext() {
        return this.iterator.hasNext();
    }

    public MultiDataSet next() {
        return this.next(this.batchSize);
    }

    public MultiDataSet next(int num) {
        if (!this.hasNext()) {
            throw new NoSuchElementException();
        } else {
            MultiDataSet out = iterator.next();

            if (this.preProcessor != null) {
                this.preProcessor.preProcess((MultiDataSet) out);
            }

            return out;
        }
    }

    private static INDArray getRange(INDArray arr, long exampleFrom, long exampleToExclusive) {
        if (arr == null) {
            return null;
        } else {
            int rank = arr.rank();
            switch (rank) {
                case 2:
                    return arr.get(new INDArrayIndex[]{NDArrayIndex.interval(exampleFrom, exampleToExclusive), NDArrayIndex.all()});
                case 3:
                    return arr.get(new INDArrayIndex[]{NDArrayIndex.interval(exampleFrom, exampleToExclusive), NDArrayIndex.all(), NDArrayIndex.all()});
                case 4:
                    return arr.get(new INDArrayIndex[]{NDArrayIndex.interval(exampleFrom, exampleToExclusive), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()});
                default:
                    throw new RuntimeException("Invalid rank: " + rank);
            }
        }
    }

    public boolean resetSupported() {
        return true;
    }

    public boolean asyncSupported() {
        return false;
    }

    public void reset() {
        if (resetSupported())
            Collections.shuffle(list);
            iterator = list.iterator();
    }

    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    public MultiDataSetPreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    public void remove() {
        throw new UnsupportedOperationException("Not supported");
    }
}


