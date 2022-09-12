package clustering;
import java.util.Random;
import java.util.stream.Stream;
import static clustering.G.*;
import java.io.IOException;

public class ParallelRandomSwap {
	static long start, end;
	static double previous_cost, current_cost;
	static int step, CI, lastCI, ncCIz, ncNegSwaps;				
	static DataPoint[] prototypes;
	static boolean accepted;
	static Random ran;
	
	static void initialize() throws IOException {
		load_dataset();
		load_centroids();
		if( INIT_GT ) load_gt();
		if( INIT_PARTITIONS ) load_partitions();
		prototypes=new DataPoint[K];
		ran=new Random();
		previous_cost=0; current_cost=0;
		CI=Integer.MAX_VALUE; lastCI=Integer.MAX_VALUE; ncCIz=0; ncNegSwaps=0;
		accepted=true;
		
		if( PARALLEL ) {
			if( OUTPUT ) {
				open("d:\\rs-files\\"+datasetName+"-RS-parallel-"+System.currentTimeMillis()+".txt");
				println("Parallel Random Swap "+datasetName+
						" N="+N+" D="+D+" K="+K+" STEPS="+T);
			}
			System.out.println("Parallel Random Swap "+datasetName+
					" N="+N+" D="+D+" K="+K+" STEPS="+T);
		}
		else {
			if( OUTPUT ) {
				open("d:\\rs-files\\"+datasetName+"-RS-serial-"+System.currentTimeMillis()+".txt");
				println("Serial Random Swap  "+datasetName+
						" N="+N+" D="+D+" K="+K+" STEPS="+T);
			}
			System.out.println("Serial Random Swap "+datasetName+
					" N="+N+" D="+D+" K="+K+" STEPS="+T);
		}
	}//initialize
	
	static void k_means() {
		for( int it=0; it<T; ++it ){
			final int IT=it;	
			//Assign data points to clusters
			Stream<DataPoint >p_stream=Stream.of( dataset );
			if( PARALLEL ) p_stream=p_stream.parallel();
			p_stream
				.map( p -> { 
					if( IT==0 ) p.saveCID();
					double md=Double.MAX_VALUE;
					for( int k=0; k<K; ++k ) {
						double d=p.distance( centroids[k] );
						if( d<md ) { md=d; p.setCID(k); }
					}
					return p; } )
				.forEach( p->{} );
			//prepare new centroids
			for( int k=0; k<K; ++k ) {
				newCentroids[k].reset();
				newCentroids[k].setCID( k );
			}
			//Update centroids
			Stream<DataPoint> c_stream=Stream.of( newCentroids );
			if( PARALLEL ) c_stream=c_stream.parallel();
			c_stream
				.map( c -> {
					for( int i=0; i<N; ++i ) {
						if( dataset[i].getCID()==c.getCID() ) c.add( dataset[i] );
					}
					c.mean();
					return c; } )
				.forEach( c->{} );
			//check for termination
			boolean end=true;
			for( int k=0; k<K; ++k ) {
				if( newCentroids[k].distance(centroids[k])>THR ) { end=false; break; }
			}	
			//copy newCentroids on to centroids
			for( int k=0; k<K; ++k ) {
				centroids[k]=new DataPoint( newCentroids[k] );
				centroids[k].setN( newCentroids[k].getN() );
			}
			if( end ) break;
		}//for( it... )		
	}//k_means 
	
	static void k_means( final int times ) {
		for( int it=0; it<times; ++it ){
			final int IT=it;	
			//assign data points to clusters
			Stream<DataPoint >p_stream=Stream.of( dataset );
			if( PARALLEL ) p_stream=p_stream.parallel();
			p_stream
				.map( p -> { 
					if( IT==0 ) p.saveCID();
					double md=Double.MAX_VALUE;
					for( int k=0; k<K; ++k ) {
						double d=p.distance( centroids[k] );
						if( d<md ) { md=d; p.setCID(k); }
					}
					return p; } )
				.forEach( p->{} );
			//prepare new centroids
			for( int k=0; k<K; ++k ) {
				centroids[k].reset();
				centroids[k].setCID( k );
			}
			//update centroids
			Stream<DataPoint> c_stream=Stream.of( centroids );
			if( PARALLEL ) c_stream=c_stream.parallel();
			c_stream
				.map( c -> {
					for( int i=0; i<N; ++i ) {
						if( dataset[i].getCID()==c.getCID() ) c.add( dataset[i] );
					}
					c.mean();
					return c; } )
				.forEach( c->{} );
		}//for( it... )		
	}//k_means 
	
	static void partition() {
		Stream<DataPoint> p_stream=Stream.of( dataset );
		if( PARALLEL ) p_stream=p_stream.parallel();
		p_stream
			.map( p -> { 
				double md=Double.MAX_VALUE;
				for( int k=0; k<K; ++k ) {
					double d=p.distance( centroids[k] );
					if( d<md ) { 
						md=d; 
						p.setCID(k); 
					}
				}
				return p; } )
			.forEach( p->{} );
	}//partition
	
	static void restore_partition() {
		Stream<DataPoint> p_stream=Stream.of( dataset );
		if( PARALLEL ) p_stream=p_stream.parallel();
		p_stream
			.map( p -> { p.restoreCID(); return p; } )
			.forEach( p->{} );
	}//restore_partition
	
	static void restore_centroids() {
		for( int k=0; k<K; ++k ) {
			centroids[k]=new DataPoint( prototypes[k] );
			centroids[k].setN( prototypes[k].getN() );
		}
	}//restore_centroids
	
	static void save_prototypes() {
		for( int k=0; k<K; ++k ) {
			prototypes[k]=new DataPoint( centroids[k] );
			prototypes[k].setN( centroids[k].getN() );
		}
	}//save_prototypes
	
	static void make_swap() {
		//random select an existing cluster - let it be j
		int j=ran.nextInt(K); //in [0,K[
		//replace centroids[j] by a randomly selected data point in the dataset
		int ix=ran.nextInt(N); //in [0,N[
		centroids[j]=new DataPoint( dataset[ix] );		
	}//make_swap
	
	static void rec_accept() {
		ncNegSwaps=0; //reset number of consecutive negative swaps
		CI=CI(centroids,prototypes); //local CI
		if( DEBUG ) {
			System.out.println("step: "+step+" nMSE="+current_cost+" CI="+CI);
		}
		if( CI==0 ) {
			if( lastCI==0 ) {
				ncCIz++; //number of consecutive CI=0 swaps
			}
			else ncCIz=1;
		}
		else ncCIz=0;
		//remember CI
		lastCI=CI;
	}//rec_accept
	
	static void rec_refuse() {
		ncNegSwaps++;
	}//rec_refuse
	
	static boolean terminate() {
		if( step==T ) return true;
		if( accepted ) return ncCIz==10; //example
		return ncNegSwaps>=500 && ncCIz>=5; //example
	}//terminate
	
	static void output() throws IOException{
		double nMSE=nMSE(), SI=silhouette();
		if( DEBUG ) {
			if( INIT_GT ) {
				if( accepted )
					System.out.println("Last step A: "+step+" nMSE="+nMSE()+" ncCIz="+ncCIz+" CI="+CI+" CIvsGT="+CI(centroids,gt));
				else
					System.out.println("Last step R: "+step+" nMSE="+nMSE()+" ncCIz="+ncCIz+" CI="+CI+" CIvsGT="+CI(centroids,gt));
			}
			else if( INIT_PARTITIONS ) {
				final_partitions();
				if( accepted )
					System.out.println("Last step A: "+step+" nMSE="+nMSE()+" ncCIz="+ncCIz+" CI="+CI+" GCI="+jGCI(iPartition,fPartition));
				else
					System.out.println("Last step R: "+step+" nMSE="+nMSE()+" ncCIz="+ncCIz+" CI="+CI+" GCI="+jGCI(iPartition,fPartition));			
			}
		}
		else 
			System.out.println("Last step: "+step+" ncCIz="+ncCIz+" CI="+CI);
		
		if( PARALLEL ) System.out.println("PET "+(end-start)+" msec");
		else System.out.println("SET "+(end-start)+" msec");

		System.out.println("nMSE: "+nMSE);
		System.out.println("Silhouette index: "+SI);
		
		if( OUTPUT ) {
			if( INIT_GT ) {
				if( accepted )
					println("Last step A: "+step+" nMSE="+nMSE+" ncCIz="+ncCIz+" CI="+CI+" CIvsGT="+CI(centroids,gt));
				else
					println("Last step R: "+step+" nMSE="+nMSE+" ncCIz="+ncCIz+" CI="+CI+" CIvsGT="+CI(centroids,gt));
			}
			else if( INIT_PARTITIONS ) {
				final_partitions();
				if( accepted )
					println("Last step A: "+step+" nMSE="+nMSE+" ncCIz="+ncCIz+" CI="+CI+" GCI="+jGCI(iPartition,fPartition));
				else
					println("Last step R: "+step+" nMSE="+nMSE+" ncCIz="+ncCIz+" CI="+CI+" GCI="+jGCI(iPartition,fPartition));				
			}
			else {
				if( accepted )
					println("Last step A: "+step+" nMSE="+nMSE+" ncCIz="+ncCIz+" CI="+CI);
				else
					println("Last step R: "+step+" nMSE="+nMSE+" ncCIz="+ncCIz+" CI="+CI);
			}
			if( PARALLEL ) println("PET "+(end-start)+" msec");
			else println("SET "+(end-start)+" msec");
			println("nMSE: "+nMSE);
			println("Silhouette index: "+SI);
			//println( String.format("Average distance to nearest centroids: %1.3f%n",ad2nc()));
			println();
			if( PERSIST ) persist_solution();
			close();
		}		
	}//output
	
	public static void main( String[] args )throws IOException{
		initialize();
		start=System.currentTimeMillis();
		partition();
		previous_cost=nMSE();
		step=1;
		for( ; step<=T; ++step ) {
			if( accepted ) save_prototypes();
			make_swap();
			k_means(5);
			current_cost=nMSE(); 
			if( current_cost<previous_cost ) {	
				accepted=true; previous_cost=current_cost;
				rec_accept(); //bookeeping
			}
			else {
				accepted=false;
				restore_partition();
				restore_centroids();
				rec_refuse(); //bookeeping
			}
			//comment the following line to always execute all the T steps
			if( terminate() ) break; //possibly stop earlier
		}//for( step... )
		k_means();
		end=System.currentTimeMillis();
		//persist_centroids();
		output();
	}//main

}//KMeans
