
module bxb_ip (    
    input wire          bxb_clock, 
    input wire          bxb_reset,

    // AMM slave
    input wire [5:0]    bxb_csr_address,    //[31:0]
    input wire [31:0]   bxb_csr_writedata, 
    input wire          bxb_csr_write,
    input wire          bxb_csr_read, 
    output wire [31:0]  bxb_csr_readdata, 
    
    // adma AMM master
    output wire [31:0]  bxb_adma_address,      
    output wire [9:0]   bxb_adma_burstcount, //[10:0]
    output wire         bxb_adma_read,
    input wire [63:0]   bxb_adma_readdata,   //[31:0]     
    input wire          bxb_adma_readdatavalid,
    input wire          bxb_adma_waitrequest, 
    
    // wdma AMM master
    output wire [31:0]  bxb_wdma_address, 
    output wire [9:0]   bxb_wdma_burstcount, //[10:0]
    output wire         bxb_wdma_read,         
    input wire [31:0]   bxb_wdma_readdata, 
    input wire          bxb_wdma_readdatavalid, 
    input wire          bxb_wdma_waitrequest, 
    
    // qdma AMM master
    output wire [31:0]  bxb_qdma_address, 
    output wire [9:0]   bxb_qdma_burstcount,
    output wire         bxb_qdma_read, 
    input wire [63:0]   bxb_qdma_readdata,
    input wire          bxb_qdma_readdatavalid,
    input wire          bxb_qdma_waitrequest, 
    
    // fdma AMM master
    output wire [31:0]  bxb_fdma_address, 
    output wire [9:0]   bxb_fdma_burstcount,

    input wire          bxb_fdma_waitrequest,
    output wire         bxb_fdma_write, 
    output wire [127:0] bxb_fdma_writedata,  //[31:0]
    
    // rdma AMM master
    output wire [31:0]  bxb_rdma_address, 
    output wire [9:0]   bxb_rdma_burstcount,

    input wire          bxb_rdma_waitrequest, 
    output wire         bxb_rdma_write,
    output wire [63:0]  bxb_rdma_writedata    //[31:0]
);

Bxb Bxb_inst ( 
    .clock(bxb_clock), 
    .reset(bxb_reset),
    .io_csrSlaveAddress(bxb_csr_address), 
    .io_csrSlaveWriteData(bxb_csr_writedata), 
    .io_csrSlaveWrite(bxb_csr_write),
    .io_csrSlaveRead(bxb_csr_read), 
    .io_csrSlaveReadData(bxb_csr_readdata), 

    .io_admaAvalonAddress(bxb_adma_address), 
    .io_admaAvalonRead(bxb_adma_read), 
    .io_admaAvalonBurstCount(bxb_adma_burstcount), 
    .io_admaAvalonWaitRequest(bxb_adma_waitrequest), 
    .io_admaAvalonReadDataValid(bxb_adma_readdatavalid),
    .io_admaAvalonReadData(bxb_adma_readdata), 

    .io_wdmaAvalonAddress(bxb_wdma_address), 
    .io_wdmaAvalonRead(bxb_wdma_read), 
    .io_wdmaAvalonBurstCount(bxb_wdma_burstcount), 
    .io_wdmaAvalonWaitRequest(bxb_wdma_waitrequest), 
    .io_wdmaAvalonReadDataValid(bxb_wdma_readdatavalid), 
    .io_wdmaAvalonReadData(bxb_wdma_readdata), 

    .io_qdmaAvalonAddress(bxb_qdma_address), 
    .io_qdmaAvalonRead(bxb_qdma_read), 
    .io_qdmaAvalonBurstCount(bxb_qdma_burstcount),
    .io_qdmaAvalonWaitRequest(bxb_qdma_waitrequest), 
    .io_qdmaAvalonReadDataValid(bxb_qdma_readdatavalid),
    .io_qdmaAvalonReadData(bxb_qdma_readdata), 

    .io_fdmaAvalonAddress(bxb_fdma_address), 
    .io_fdmaAvalonBurstCount(bxb_fdma_burstcount),
    .io_fdmaAvalonWaitRequest(bxb_fdma_waitrequest),
    .io_fdmaAvalonWrite(bxb_fdma_write), 
    .io_fdmaAvalonWriteData(bxb_fdma_writedata), 
    
    .io_rdmaAvalonAddress(bxb_rdma_address), 
    .io_rdmaAvalonBurstCount(bxb_rdma_burstcount),
    .io_rdmaAvalonWaitRequest(bxb_rdma_waitrequest), 
    .io_rdmaAvalonWrite(bxb_rdma_write),
    .io_rdmaAvalonWriteData(bxb_rdma_writedata)
);

endmodule
