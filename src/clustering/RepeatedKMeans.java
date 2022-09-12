package clustering;
import java.util.stream.Stream;
import static clustering.G.*;
import java.io.IOException;
public class RepeatedKMeans {
	
    static boolean termination() {
    	boolean end=false;
    	if( it>T ) end=true;
    	else {
    		end=true;
    		for( int i=0; i<K; ++i )
    			if( newCentroids[i].distance(centroids[i])>THR ) {
    				end=false;
    				break;
    			}
    	}
    	//copy newCentroids on to centroids
    	for( int i=0; i<K; ++i ) {
    		centroids[i].setCoord( newCentroids[i].getCoord() );
    		centroids[i].setN( newCentroids[i].getN() );
    	}
    	return end;
    }//termination
    
	static void final_partitions() {
		for( int i=0; i<N; ++i ) {
			int k=dataset[i].getCID();
			fPartition[k].add(i);
		}
	}//final_partitions
	
	public static void main( String[] args )throws IOException{
		if( PARALLEL )
			System.out.println("Parallel Repeated K-Means "+datasetName+
				" N="+N+" D="+D+" K="+K);
		else
			System.out.println("Serial Repeated K-Means "+datasetName+
				" N="+N+" D="+D+" K="+K);
		load_dataset();
		
		double nMSE=Double.MAX_VALUE;
		int CI=0, GCI=0, success=0, numIT=0;
		long total=0;
		
		if( OUTPUT ) {
			if( PARALLEL ) {
				open("d:\\rkm-files\\"+datasetName+"-RKM-parallel-"+System.currentTimeMillis()+".txt");
				println("Parallel RKM "+datasetName+" N="+N+" D="+D+" K="+K+" RUNS="+RUNS+" STEPS="+T);			
			}
			else {
				open("d:\\rkm-files\\"+datasetName+"-RKM-serial-"+System.currentTimeMillis()+".txt");
				println("Serial RKM "+datasetName+" N="+N+" D="+D+" K="+K+" RUNS="+RUNS+" STEPS="+T);							
			}
		}

		for( int run=0; run<RUNS; ++run ) {
			load_centroids();
			if( INIT_GT ) load_gt();
			else if( INIT_PARTITIONS ) load_partitions();
			it=0;
			long start=System.currentTimeMillis();
			do{
				//assign data points to clusters
				Stream<DataPoint> p_stream=Stream.of( dataset );
				if( PARALLEL ) p_stream=p_stream.parallel();
				p_stream
					.map( p -> { 
						double md=Double.MAX_VALUE;
						for( int k=0; k<K; ++k ) {
							double d=p.distance( centroids[k] );
							if( d<md ) { md=d; p.setCID(k); }
						}
						return p; } )
					.forEach( p->{} );
			
				//prepare newCentroids
				for( int i=0; i<K; ++i ) {
					newCentroids[i].reset();
					newCentroids[i].setCID( i );
				}
			
				//update centroids
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
				it++;
			}while( !termination() );
			long end=System.currentTimeMillis();
			total=total+(end-start);
			numIT=numIT+it;

			double cost=nMSE();
			if( cost<nMSE ) {
				nMSE=cost;
				if( OUTPUT ) {
					println("nMSE="+nMSE);
					if( PERSIST ) persist_centroids();
				}
			}
			
			if( INIT_GT ) {
				int ci=CI(centroids,gt);
				CI=CI+ci;
				if( ci==0 ) success++;
			}
			else if( INIT_PARTITIONS ) {
				final_partitions();
				int gci=jGCI(iPartition,fPartition);
				GCI=GCI+gci;
				if( gci==0 ) success++;
			}	
		}//for( run... )
		
		if( PARALLEL )
			System.out.println("PET "+total+" msec");
		else
			System.out.println("SET "+total+" msec");

		System.out.println("minimum nMSE: "+nMSE);
		System.out.println("average number of iterations per run="+((double)numIT/RUNS));
		System.out.println("success rate="+((double)success/RUNS));
		if( INIT_GT ) {
			System.out.println("average CI="+((double)CI/RUNS));
			System.out.println("average rel-CI="+( (double)CI/RUNS)/K );
		}
		else if( INIT_PARTITIONS ) {
			System.out.println("average GCI="+((double)GCI/RUNS));
			System.out.println("average rel-GCI="+( (double)GCI/RUNS)/K );
		}		

		if( OUTPUT ) {
			println("minimum nMSE: "+nMSE);
			println("average number of iterations per run="+((double)numIT/RUNS));
			println("success rate="+((double)success/RUNS));
			if( INIT_GT ) {
				println("average CI="+((double)CI/RUNS));
				println("average rel-CI="+( (double)CI/RUNS)/K );
			}
			else if( INIT_PARTITIONS ) {
				println("average GCI="+((double)GCI/RUNS));
				println("average rel-GCI="+( (double)GCI/RUNS)/K );
			}
			println();
			close();
		}

	}//main

}//RKM
