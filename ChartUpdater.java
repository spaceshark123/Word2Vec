import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.chart.title.TextTitle;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.ui.ApplicationFrame;

import javax.swing.*;
import java.awt.*;
import java.text.DecimalFormat;

public class ChartUpdater implements NeuralNetwork.TrainingCallback {
    private XYSeries trainAccuracySeries;
    private XYSeries testAccuracySeries;
    private JFreeChart chart;
    private TextTitle caption;
    private int epochs;

    public ChartUpdater(int epochs) {
        trainAccuracySeries = new XYSeries("Train Accuracy");
        testAccuracySeries = new XYSeries("Test Accuracy");

        this.epochs = epochs;
        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(trainAccuracySeries);
        dataset.addSeries(testAccuracySeries);
        caption = new TextTitle("Current Epoch: 0, Current Batch: 0, Current Accuracy: 0.0%");
        chart = ChartFactory.createXYLineChart(
                "Accuracy over Epochs",
                "Epoch",
                "Accuracy",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false);

        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
        renderer.setSeriesPaint(0, Color.RED);
        XYLineAndShapeRenderer renderer2 = new XYLineAndShapeRenderer();
        renderer2.setSeriesPaint(0, Color.BLUE);
        plot.setRenderer(0, renderer);
        plot.setRenderer(1, renderer2);
        plot.getRangeAxis().setRange(0, 100);

        NumberAxis domainAxis = (NumberAxis) plot.getDomainAxis();
        domainAxis.setTickUnit(new NumberTickUnit(1));

        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        DecimalFormat percentFormat = new DecimalFormat("0.0%");
        percentFormat.setMultiplier(1);
        rangeAxis.setNumberFormatOverride(percentFormat);

        chart.addSubtitle(caption);

        ChartPanel panel = new ChartPanel(chart);
        panel.setPreferredSize(new Dimension(800, 600));
        JFrame frame = new JFrame();
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setAlwaysOnTop(true);
        frame.setVisible(true);
    }

    @Override
    public void onEpochUpdate(int epoch, int batch, double progress, double trainAccuracy, double testAccuracy) {
        trainAccuracySeries.add(progress, trainAccuracy);
        if(testAccuracy != -1) {
            testAccuracySeries.add(progress, testAccuracy);
        }
        // Update caption
        if(epoch <= epochs) {
            caption.setText(String.format("Current Epoch: %d, Current Batch: %d, Current Accuracy: %f%%", epoch, batch, trainAccuracy));
            chart.fireChartChanged();
        }
    }
}