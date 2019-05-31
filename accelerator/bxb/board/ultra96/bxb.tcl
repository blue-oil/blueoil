## Customisable variables
set root "."

#start_gui
create_project project_1 $root/project_1 -part xczu3eg-sbva484-1-e

ipx::infer_core -vendor user.org -library user -taxonomy /UserIP $root/rtl
ipx::edit_ip_in_project -upgrade true -name edit_ip_project -directory $root/project_1/project_1.tmp $root/rtl/component.xml

ipx::current_core $root/rtl/component.xml
update_compile_order -fileset sources_1
ipx::infer_bus_interfaces xilinx.com:interface:avalon_rtl:1.0 [ipx::current_core]

ipx::add_memory_map bxb_csr [ipx::current_core]
set_property slave_memory_map_ref bxb_csr [ipx::get_bus_interfaces bxb_csr -of_objects [ipx::current_core]]
ipx::add_address_block Reg [ipx::get_memory_maps bxb_csr -of_objects [ipx::current_core]]
set_property BASE_ADDRESS {0} [ipx::get_address_block Reg [ipx::get_memory_maps bxb_csr -of_objects [ipx::current_core]]]
set_property RANGE {4096} [ipx::get_address_block Reg [ipx::get_memory_maps bxb_csr -of_objects [ipx::current_core]]]

ipx::add_address_space bxb_adma [ipx::current_core]
set_property master_address_space_ref bxb_adma [ipx::get_bus_interfaces bxb_adma -of_objects [ipx::current_core]]
set_property RANGE {2G} [ipx::get_address_spaces bxb_adma]
set_property WIDTH {64} [ipx::get_address_spaces bxb_adma]

ipx::add_address_space bxb_fdma [ipx::current_core]
set_property master_address_space_ref bxb_fdma [ipx::get_bus_interfaces bxb_fdma -of_objects [ipx::current_core]]
set_property RANGE {2G} [ipx::get_address_spaces bxb_fdma]
set_property WIDTH {128} [ipx::get_address_spaces bxb_fdma]

ipx::add_address_space bxb_qdma [ipx::current_core]
set_property master_address_space_ref bxb_qdma [ipx::get_bus_interfaces bxb_qdma -of_objects [ipx::current_core]]
set_property RANGE {2G} [ipx::get_address_spaces bxb_qdma]
set_property WIDTH {64} [ipx::get_address_spaces bxb_qdma]

ipx::add_address_space bxb_rdma [ipx::current_core]
set_property master_address_space_ref bxb_rdma [ipx::get_bus_interfaces bxb_rdma -of_objects [ipx::current_core]]
set_property RANGE {2G} [ipx::get_address_spaces bxb_rdma]
set_property WIDTH {64} [ipx::get_address_spaces bxb_rdma]

ipx::add_address_space bxb_wdma [ipx::current_core]
set_property master_address_space_ref bxb_wdma [ipx::get_bus_interfaces bxb_wdma -of_objects [ipx::current_core]]
set_property RANGE {2G} [ipx::get_address_spaces bxb_wdma]
set_property WIDTH {32} [ipx::get_address_spaces bxb_wdma]

set_property core_revision 2 [ipx::current_core]
ipx::update_source_project_archive -component [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]

exit